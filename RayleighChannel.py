import numpy as np
class rayleigh_multipath(object):
    """a multipath channel with Rayleigh Fading and AWGN"""
    def __init__(self, sigma_awgn, sigma_rayleigh, pdp):
        self.sigma_awgn = sigma_awgn
        self.sigma_rayleigh = sigma_rayleigh
        self.pdp = np.array(pdp)
        self.l = self.pdp.size - 1
        self.update_cir()

    def update_cir(self):
        """Generare a new CIR from the PDP with Rayleigh Fading"""
        self.cir = np.sqrt(np.array(self.pdp))
        randray = np.random.rayleigh(self.sigma_rayleigh,self.cir.size)
        self.cir = self.cir * randray

    def awgn(self, symbols):
        """add Gaussian White Noise"""
        #real_noise = np.random.rand(symbols.size)
        #imag_noise = np.random.rand(symbols.size)
        noise = np.random.rand(symbols.size)
        return symbols + self.sigma_awgn * noise

    def apply_cir(self,symbols):
        """convolve the symbols with cir"""
        if self.l != 0:
            self.old_symbols = symbols[-self.l :]
        #apply the cir
        symbols = np.convolve(symbols, self.cir)
        return symbols