import torch
import torch.nn as nn
import snntorch as snn

class RFSynaptic(snn.RSynaptic):
    # This class implements a resonant frequency model that uses RSynaptic's syn to maintain the secondary component of 
    # the resonant state for implementation efficiency. 
    def __init__(
        self,
        alpha=0.0,
        beta=0.0,
        *args,
        T_min = 3,
        T_max = 100,
        rho=0.95,
        spike_damp=0.2,
        refrac_strength=0.5,
        refrac_decay=0.9,
        period_spacing='log',
        **kwargs
    ):
        super().__init__(alpha, beta, *args, **kwargs)


        # sample periods to decide what frequencies to use for the oscillators
        # defaults to log because (in principle) it gives a greater coverage of short, medium, and long periods 
        # than linear space. I did not end up studying behaviour under linspace regime
        if period_spacing == 'log':
            periods = torch.logspace(
                torch.log10(torch.tensor(float(T_min))),
                torch.log10(torch.tensor(float(T_max))),
                self.linear_features
            )
        elif period_spacing == 'linear':
            periods = torch.linspace(T_min,T_max, self.linear_features)
        omega = 2 * torch.pi / periods

        omega = torch.as_tensor(omega)
        rho = torch.as_tensor(rho)

        self.register_buffer("spike_damp", torch.as_tensor(spike_damp))
        self.register_buffer("refrac_strength", torch.as_tensor(refrac_strength))
        self.register_buffer("refrac_decay", torch.as_tensor(refrac_decay))
        self.register_buffer("omega", omega)
        self.register_buffer("rho", rho)
        self.register_buffer("refrac", torch.zeros(0), False)

    def reset_mem(self):
        spk, syn, mem = super().reset_mem()
        self.refrac = torch.zeros_like(mem, device=mem.device)
        return spk, syn, mem, self.refrac

    def _base_state_function(self, input_):
        # override the base state function to implement the resonant frequency dynamics

        # recurrent reservoir input
        current = input_ + self.recurrent(self.spk)

        # mem = u, syn = v
        u_old = self.mem
        v_old = self.syn

        c = torch.cos(self.omega)
        s = torch.sin(self.omega)

        damp = 1.0 - self.spike_damp * self.spk
        rho_eff = self.rho.clamp(0, 1) * damp

        u_base = rho_eff * (c * u_old - s * v_old) + current
        v_base = rho_eff * (s * u_old + c * v_old)

        # RSynaptic expects: return syn, mem
        # so return v, u
        return v_base, u_base
    
    def forward(self, input_, spk=None, syn=None, mem=None, refrac=None):
        # override the forward method to implement the resonant frequency dynamics with adaptive refractory period 
        # for robustness
        if spk is not None:
            self.spk = spk
        if syn is not None:
            self.syn = syn
        if mem is not None:
            self.mem = mem
        if refrac is not None:
            self.refrac = refrac

        if self.spk.shape != input_.shape:
            self.spk = torch.zeros_like(input_, device=input_.device)

        if self.syn.shape != input_.shape:
            self.syn = torch.zeros_like(input_, device=input_.device)

        if self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_, device=input_.device)

        if not hasattr(self, "refrac") or self.refrac.shape != input_.shape:
            self.refrac = torch.zeros_like(input_, device=input_.device)

        # RF dynamics happen here
        self.reset = self.mem_reset(self.mem)
        self.syn, self.mem = self.state_function(input_)
        # syn = v_base, mem = u_base

        if self.state_quant:
            self.syn = self.state_quant(self.syn)
            self.mem = self.state_quant(self.mem)

        # adaptive threshold: fire if mem > threshold + refrac
        if self.inhibition:
            self.spk = self.fire_inhibition(
                self.mem.size(0),
                self.mem - self.refrac
            )
        else:
            self.spk = self.fire(self.mem - self.refrac)

        # update refractory state using the new spike
        self.refrac = (
            self.refrac_decay * self.refrac
            + self.refrac_strength * self.spk
        )

        # spike-triggered damping after spike
        damp = 1.0 - self.spike_damp * self.spk
        self.mem = damp * self.mem
        self.syn = damp * self.syn

        if self.output:
            return self.spk, self.syn, self.mem, self.refrac
        elif self.init_hidden:
            return self.spk
        else:
            return self.spk, self.syn, self.mem, self.refrac
