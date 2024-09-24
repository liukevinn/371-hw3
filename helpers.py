from autograd import numpy as anp
from autograd import grad
import numpy as snp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size


def fixed_axis_figure(box, scale=1., axis='on', h_pads=None, v_pads=None):
    if h_pads is None:
        h_pads = [0.7, 0.1] if axis == 'on' else [0, 0]
    if v_pads is None:
        v_pads = [0.5, 0.1] if axis == 'on' else [0, 0]
    ax_h_size, ax_v_size = scale * (box[1] - box[0]), scale * (box[3] - box[2])
    fig_h_size, fig_v_size = h_pads[0] + ax_h_size + h_pads[1], v_pads[0] + ax_v_size + v_pads[1]
    fig = plt.figure(figsize=(fig_h_size, fig_v_size))
    horizontal = [Size.Fixed(h_pads[0]), Size.Fixed(ax_h_size), Size.Fixed(h_pads[1])]
    vertical = [Size.Fixed(v_pads[0]), Size.Fixed(ax_v_size), Size.Fixed(v_pads[1])]
    d = Divider(fig, (0, 0, 1, 1), horizontal, vertical,
                aspect=False)
    ax = fig.add_axes(d.get_position(), axes_locator=d.new_locator(nx=1, ny=1))
    plt.xlim(box[:2])
    plt.ylim(box[2:])
    plt.axis(axis)
    return fig, ax


class Stepper:
    def __init__(self, f, z0, alpha, history=False, **kwargs):
        self.f = f
        self.g = grad(f)
        self.alpha = alpha
        self.count = 0
        self.d = len(z0)
        self.n = len(kwargs['indices']) if 'indices' in kwargs else 1
        self.values_only = self.d > 2
        self.history = [] if history else None
        self.fz0 = self.f(z0, **kwargs)
        self.gz0 = self.g(z0, **kwargs)
        self.save_history(self.fz0, z0)

    def save_history(self, fz, z):
        self.count += self.d * self.n
        if self.history is not None:
            self.history.append(fz if self.values_only else (fz, snp.copy(z)))

    def __call__(self, z_old, **kwargs):
        gz = self.g(z_old, **kwargs)
        s = - self.alpha * gz
        z = z_old + s
        fz = self.f(z, **kwargs)
        self.save_history(fz, z)
        return s, z, fz, gz

    def show_history(self):
        if self.history is not None:
            fz = self.history if self.values_only \
                else [h[0] for h in self.history]
            plt.figure(figsize=(6, 3), tight_layout=True)
            plt.plot(fz)
            plt.xlabel('iteration')
            plt.ylabel(r'$f(z)$')
            plt.title('{} scalar derivatives computed'.format(self.count))
            plt.show()
            if not self.values_only:
                z = snp.array([h[1] for h in self.history])
                margin = 0.1
                box = [snp.min(z[:, 0]) - margin, snp.max(z[:, 0] + margin),
                       snp.min(z[:, 1]) - margin, snp.max(z[:, 1]) + margin]
                bw, bh = box[1] - box[0], box[3] - box[2]
                if bw > bh:
                    h_center = (box[2] + box[3]) / 2
                    box[2] = h_center - bw / 2
                    box[3] - h_center + bw / 2
                else:
                    w_center = (box[0] + box[1]) / 2
                    box[0] = w_center - bh / 2
                    box[1] = w_center + bh / 2
                bw, bh = box[1] - box[0], box[3] - box[2]
                width, height, samples = 5, 5, 101
                scale = min([width / bw, height / bh])
                xx = snp.linspace(box[0], box[1], samples)
                yy = snp.linspace(box[2], box[3], samples)
                x = snp.array(snp.meshgrid(xx, yy))
                try:
                    fx = self.f(x)
                    fig, ax = fixed_axis_figure(box, scale=scale)
                    c = ax.contour(x[0], x[1], fx, levels=20,
                                   linewidths=1, colors='gray')
                    plt.plot(z[:, 0], z[:, 1], lw=2, color='r')
                    plt.plot(z[0, 0], z[0, 1], 'xr', label='start')
                    plt.plot(z[-1, 0], z[-1, 1], 'or', label='finish')
                    plt.legend()
                    plt.xlabel(r'x_0')
                    plt.xlabel(r'x_1')
                    plt.draw()
                except ValueError:
                    # f is not suitably vectorized; give up
                    pass
