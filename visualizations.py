import torch
from torchvision.utils import make_grid

import numpy as np
import matplotlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 300  # noqa: E702
import os

from utils import get_conditional_limits


def plot_visual_checks(model, device, sample_batch, odirs, dataloader, mparams, epoch):
    plot_reconstructions(model, device, sample_batch, ["Training image reconstructions", "Test image reconstructions"],
                         os.path.join(odirs['recons'], f'Reconstructions_epoch{epoch}.png'), dataloader)
    plot_local_conditional_effect(sample_batch[0], model, device, os.path.join(odirs['recons'], f'Conditional_Reconstructions_epoch{epoch}.png'), dataloader)
    plot_conditional_effect(model, device, os.path.join(odirs['grid_recons'], f'Epoch{epoch}'), mparams["latent_dim"], dataloader)


def plot_verification_plots(ver_results, images_dir, classes, test_transform_names, bi):
    print(ver_results)
    plot_ld_dists_distr(ver_results['deltas'], os.path.join(images_dir, f'latent_space_deltas_{bi}.png'))
    # plot_ld_cond_distr(ver_results['ldim_eps'], np.sum(ver_results['tested']), os.path.join(images_dir, f'latent_space_cond_eps_{bi}.png'))
    plot_ver_results(ver_results['total'], ver_results['tested'], ver_results['verified'], classes, test_transform_names, os.path.join(images_dir, f'ver_results_{bi}.png'))
    plot_ver_times(ver_results['ver_times'], os.path.join(images_dir, f'ver_times_{bi}.png'))


def plot_verification_results(model, device, result_imgs, odir, dataloader, bi):
    imgset_titles = ["Image1", "Interpolated reconstructions", "Image2"]
    plot_interp_reconstuctions(model, device, result_imgs['safe'], imgset_titles, os.path.join(odir, f'ver_recons_{bi}.png'), dataloader)
    plot_interp_reconstuctions(model, device, result_imgs['unsafe'], imgset_titles, os.path.join(odir, f'unver_recons_{bi}.png'), dataloader, result_imgs['ceg_z'])
    result_imgs['safe'], result_imgs['unsafe'], result_imgs['ceg_z'] = [], [], []


def plot_images(images_batches, titles, image_path, row_labels=None, num_rows=None, wratio=None):
    assert len(images_batches) == len(titles)
    wratio = wratio or [1]*len(images_batches)
    fig, axes = plt.subplots(ncols=len(images_batches), figsize=(8*len(images_batches), 10), gridspec_kw={'width_ratios': wratio})
    for i, batch in enumerate(images_batches):
        batch = batch.to("cpu")
        # assert batch.dim == 4 and [NxCxHxW]
        ax = axes[i] if len(images_batches) > 1 else axes
        if row_labels is not None:
            nrows = len(row_labels)
            ax.set_xticks([])
            ax.set_yticks([batch[0].shape[1]*(2*i+1)/2 + 2*i+1 for i in range(nrows)], labels=row_labels)
        elif num_rows is not None:
            ax.axis("off")
            nrows = num_rows[i]
        else:
            ax.axis("off")
            nrows = int(np.sqrt(batch.shape[0]))

        ax.imshow(
            np.transpose(
                make_grid(batch, nrow=int(batch.shape[0]/nrows), padding=2, normalize=True).cpu(),
                (1, 2, 0)
            )
        )
        ax.set_title(titles[i])
    fig.tight_layout()
    plt.savefig(image_path)
    plt.close("all")


def plot_embeddings(zs, z_noisys, y, conditional_dims, num_classes, epoch, image_dir, sw=None):
    z, z_noisy = torch.as_tensor(np.array(zs)), torch.as_tensor(np.array(z_noisys))
    if sw is not None:
        y_torch = torch.as_tensor(np.array(y))
        sw.add_embedding(torch.cat((z, z_noisy)), metadata=torch.cat((y_torch, y_torch)), global_step=epoch)
    # assert z.dim() == 4
    ld, nrows = z.shape[1], 5
    cmap = plt.cm.get_cmap('hsv', np.max((num_classes, 10)))
    colors = [cmap(n) for n in np.linspace(0.5, 1, num_classes)]
    legend_handles = [matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=i, markerfacecolor=colors[i], markersize=10) for i in range(num_classes)]

    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(30, 6*nrows))
    fig.tight_layout()
    for ai in range(-4, 6, 2):  # to be able to plot both conditional (2) and continuous latent dims
        c_ld = conditional_dims+ai
        if c_ld < -1:
            continue
        if c_ld >= ld:
            break
        ax = axes[int((ai+4)/2)] if nrows > 1 else axes
        for i, (zi, tlabel) in enumerate(zip([z, z_noisy], ["", " (used for GAN)"])):
            if len(zi) == 0:
                continue
            z1 = zi[:, c_ld]
            if c_ld == -1:
                z1 = torch.zeros(zi[:, c_ld+1].shape)
            ax[i].scatter(z1, zi[:, c_ld+1], s=8, c=[colors[yy] for yy in y])
            ax[i].set_title(f'Latent vectors {c_ld} vs {c_ld+1}{tlabel}')
            ax[i].legend(handles=legend_handles, loc='lower right')
        ul = 1.01 if c_ld < conditional_dims else 3.5
        ll = 0.05 if c_ld < conditional_dims else -3.5
        if c_ld > -1:
            ax[2].hist(z[:, c_ld].cpu().numpy(), bins=50, density=True, range=(ll, ul))
        ax[3].hist(z[:, c_ld+1].cpu().numpy(), bins=50, density=True, range=(ll, ul))
    image_path = os.path.join(image_dir, f'Embeddings_epoch{epoch}.png')
    plt.savefig(image_path)
    plt.close("all")


def plot_reconstructions(model, device, images_batch, titles, image_path, dataloader):
    recons_batches = []
    for i, batch in enumerate(images_batch):
        with torch.no_grad():
            recons_batch = model(batch.to(device))[-1]
        recons_batches.append(dataloader.denormalize(recons_batch))
    classes, = dataloader.get_dataset_params(["classes"])
    plot_images(recons_batches, titles, image_path, row_labels=classes)


def plot_local_conditional_effect(sample_batch, model, device, image_path, dataloader):
    """ Draws conditional variations (around setpoints as per conditional_loss) of each class image """
    classes, conditional_dims = dataloader.get_dataset_params(["classes", "conditional_ldims"])
    if conditional_dims == 0:
        return
    imgs_per_class, n = int(sample_batch.shape[0]/len(classes)), 5
    conditional_limits, _ = get_conditional_limits(dataloader)

    oimages, zs = [], []
    for j in range(len(classes)):
        img = sample_batch[j*imgs_per_class+imgs_per_class-1]
        oimages.append(img)
        _, z, _ = model(img.unsqueeze(0).to(device), only_gen_z=True)
        zs.append(z)  # the first is exact reconstruction
        z_clean = clean_z(z, conditional_dims)
        zs.append(z_clean)  # second one is without any transform/attribute
        for i in range(conditional_dims):
            climits = conditional_limits[i]
            for climit in climits:
                for cvalue in np.linspace(climit[0], climit[1], n):  # 0.75, 1.2
                    z_i = z.clone()
                    z_i[0, i] += cvalue
                    zs.append(z_i)
    z = torch.stack((zs), dim=0)
    with torch.no_grad():
        images = dataloader.denormalize(model.decoder(z.squeeze(1).to(device)))
    oimages = torch.stack(oimages)
    plot_images([oimages, images], ["Original images", "Conditional transformations of classes"], image_path, row_labels=classes, wratio=[1, n*conditional_dims])


def plot_conditional_effect(model, device, image_path, latent_dims, dataloader):
    """ Produces 2 plots:
        1. Has subplots showing reconstructions from varying a continuous with each row having one active conditional dimension
        2. Plots each row transforming from one attribute/conditional to another (continuous dimensions being generic and fixed)
    """
    attributes, conditional_dims = dataloader.get_dataset_params(["attributes", "conditional_ldims"])
    if conditional_dims == 0:
        return
    conditional_limits, conditional_labels = get_conditional_limits(dataloader)
    n = 5

    if latent_dims > conditional_dims:  # continuous are present
        num_subplots = np.min((latent_dims-conditional_dims, 5))  # one subplot for each continuous dimension
        images, titles = [], []
        for j in range(num_subplots):
            zs = []
            for i in range(conditional_dims):
                for y in np.linspace(-2., 2., n):
                    z_np = [0]*latent_dims
                    z_np[i] = np.mean(conditional_limits[i][-1])
                    z_np[conditional_dims+j] = y  # variations in continuous dimensions for a given conditional dimension with any setpoint (constant for row)
                    zs.append(z_np)
            z = torch.Tensor(zs).to(device)
            with torch.no_grad():
                images.append(dataloader.denormalize(model.decoder(z)))
            titles.append(f'Varying cont. var {j}')
        plot_images(images, titles, f'{image_path}_continuous_reconstructions.png', row_labels=conditional_labels)

    if conditional_dims < 2:
        return
    # plot with each row showing a conditional class transforming from one to the next one
    n, zs, row_labels, images = 5, [], [], []
    for i in range(conditional_dims-1):
        climits = conditional_limits[i]
        for j, climit in enumerate(climits):
            csteps = np.linspace(climit[1], climit[0], n)  # reduce in current active conditional dim in n steps
            nlimits = climits[j+1] if j < len(climits)-1 else conditional_limits[i+1][0]
            nsteps = np.linspace(*nlimits, n)  # increase in next conditional dim in n steps
            # csteps = np.linspace(1.2, 0, n)  # reduce in current active conditional dim in n steps
            # nsteps = np.linspace(0, 1.2, n)  # increase in next conditional dim in n steps
            for dc, dn in zip(csteps, nsteps):
                z_np[i], z_np[i+1] = dc, dn
                zs.append(z_np)
        row_labels.append(f'{conditional_labels[i]} to {conditional_labels[i+1]}')
    z = torch.Tensor(zs).to(device)
    with torch.no_grad():
        x_hat = dataloader.denormalize(model.decoder(z))
    plot_images([x_hat], ["Conditional transformations"], f'{image_path}_conditional_transformations.png', row_labels=row_labels)


# verification script plots
def plot_local_conditionals(image_batches, z_ceg_batches, model, device, image_path, dataloader):
    """ Plots conditionals and counterexamples found after verification """
    assert len(image_batches) == len(z_ceg_batches), print(len(image_batches), len(z_ceg_batches))
    if len(image_batches) == 0:
        return
    set_pts = [0., 0.25, 0.5, 1., 1.5, 2.5]

    oimages, zs, zs_ceg_nn = [], [], []
    model = model.to(device)
    for img, ceg in zip(image_batches, z_ceg_batches):
        img = img.to(device)
        dim_ind, ceg_eps, z_ceg = ceg
        oimages.append(img)
        with torch.no_grad():
            _, z, _ = model(img, only_gen_z=True)
        for ei, eps in enumerate(set_pts):
            if eps <= ceg_eps and (ei == len(set_pts)-1 or ceg_eps < set_pts[ei+1]):
                z_ceg = z_ceg.to(device)
                zs.append(z_ceg); zs_ceg_nn.append(z_ceg)
            z_i = z.clone()
            z_i[0, dim_ind] += eps
            zs.append(z_i)
    z, z_ceg_nn = torch.stack((zs), dim=0), torch.stack((zs_ceg_nn), dim=0)
    with torch.no_grad():
        images = dataloader.denormalize(model.decoder(z.squeeze(1)))
        cegs = dataloader.denormalize(model.decoder(z_ceg_nn.squeeze(1)))
    oimages = torch.concat(oimages, axis=0)
    assert cegs.shape == oimages.shape and images.dim() == 4 and cegs.dim() == 4, print(images.shape, oimages.shape, cegs.shape)
    plot_images([oimages, images, cegs], ["Original images", "Conditional transformations of classes", "Counterexample"], image_path, num_rows=[oimages.shape[0]]*3, wratio=[1, len(set_pts)+1, 1])


def plot_ld_dists_distr(deltas_hists, images_path):
    # matplotlib.rc('font', **{'family': 'normal', 'size': 10})
    
    fig, axes = plt.subplots(nrows=len(deltas_hists), ncols=len(list(deltas_hists.values())[0]), figsize=(50, 5), gridspec_kw={'hspace': 0.9, 'wspace': 0.4})
    fig.tight_layout()
    bins = np.linspace(0, 3.5, 11)
    for i, (k, v) in enumerate(deltas_hists.items()):
        for j, dim_deltas in enumerate(v):
            if dim_deltas is None:
                continue
            axes[i, j].bar(bins[:-1], dim_deltas, 0.3, align='edge', color=(0.2, 0.4, 0.6, 0.6))
            axes[i, j].set_title(f'{k}: latent dim {j}')
            axes[i, j].set_xlim(-0, 3.51)
    plt.suptitle("Distribution of delta between original and re-encoded latent vector")
    plt.savefig(images_path)
    plt.close("all")


def plot_ld_cond_distr(epses, num_tested, images_path):
   # matplotlib.rc('font', **{'family': 'normal', 'size': 12})

    fig, axes = plt.subplots(ncols=len(epses), figsize=(60, 5), gridspec_kw={'hspace': 1.6, 'wspace': 0.4})
    fig.tight_layout()
    w = 0.25
    for i, dim_epses in enumerate(epses):
        ax = axes if len(epses) == 1 else axes[i]
        rect = ax.bar(np.array(list(dim_epses.keys()))-0.5, list(dim_epses.values()), w, align='edge', color=(0.2, 0.4, 0.6, 0.6))
        ax.bar_label(rect, label_type='center')
        ax.set_ylim([0, num_tested])
        if len(dim_epses.keys()) > 0:
            ax.set_xlim([0, list(dim_epses.keys())[-1]+0.01])
        ax.set_ylabel("# verified")
        ax.set_xlabel("eps")
        ax.set_title(f'latent dim {i}')
    plt.suptitle("Number of examples that could be verified against epsilons")
    plt.savefig(images_path)
    plt.close("all")


def plot_ver_results(total_pc, tested_pc, verified_pt_pc, classes, test_transforms, images_path):
    # matplotlib.rc('font', **{'family': 'normal', 'size': 8})

    cw = 13*len(classes)
    xx = np.arange(0, cw*(len(classes)+1), cw)
    x = xx[:-1]
    w = cw/(len(verified_pt_pc)+2)

    fig = plt.figure(figsize=(2*len(classes), 5))
    fig.tight_layout()
    # colors = plt.cm.get_cmap('tab20b').colors[2:]
    # colors += colors
    cmap = plt.cm.get_cmap('Blues', np.max((len(test_transforms), 10)))
    colors = [cmap(n) for n in np.linspace(0.5, 1, len(test_transforms))]
    for i, tot in enumerate(total_pc):
        plt.hlines(y=total_pc[i], xmin=xx[i]-w, xmax=xx[i+1]-2*w, color='black', linewidth=4)
        plt.hlines(y=tested_pc[i], xmin=xx[i]-w, xmax=xx[i+1]-2*w, colors='cyan', linestyles='--', linewidth=2)
    
    for i, t_data in enumerate(verified_pt_pc):
        rect = plt.bar(x+w*i, t_data, w, label=test_transforms[i], align='center', color=colors[i], edgecolor='black')
        plt.bar_label(rect, label_type='center')

    plt.ylim(0, np.max(tested_pc)+10)
    plt.ylabel('# images')
    plt.title('# images per class and transforms', pad=20)
    plt.xticks(x+w*(len(verified_pt_pc)/2-1) + w/2, classes)
    plt.legend(bbox_to_anchor=(1.08, 1), loc='upper right', borderaxespad=0.)
    plt.savefig(images_path)
    plt.close("all")


def plot_interp_reconstuctions(model, device, images_batch, titles, image_path, dataloader, z_cegs=None):
    if len(images_batch) == 0:
        return
    if z_cegs is not None:
        assert len(images_batch) == len(z_cegs), print(len(images_batch), len(z_cegs))
    
    def recons_from_z(z):
        x = dataloader.denormalize(model.decoder(z.to(device)))
        return x

    model = model.to(device)
    n, imgs1, recons, imgs2, cegs = 5, [], [], [], []
    for i, xs in enumerate(images_batch):
        imgs1.append(xs[0])
        imgs2.append(xs[1])

        if z_cegs is not None:
            cegs.extend(recons_from_z(z_cegs[i]))

        _, zs, _ = model(xs.to(device), only_gen_z=True)
        z = torch.stack([zs[0] + (zs[1] - zs[0])*t for t in np.linspace(0, 1, n)])
        recons.extend(recons_from_z(z))

    imgs1 = torch.stack(imgs1).to("cpu"); imgs2 = torch.stack(imgs2).to("cpu"); recons = torch.stack(recons).to("cpu")  # noqa: E702
    images = [imgs1, recons, imgs2]
    if z_cegs is not None:
        images.append(torch.stack(cegs).to("cpu"))
        titles.append("Counterexample")
    plot_images(images, titles, image_path, num_rows=[len(images_batch)] * len(titles))


def plot_ver_times(ver_times, path):
    fig = plt.figure(figsize=(4, 5))
    fig.tight_layout()
    bins = np.linspace(0, 30, 10)
    for key, value in ver_times.items():
        plt.hist(value, bins, alpha=0.4, label=key)
    plt.legend(loc='upper right')
    plt.savefig(path)
    plt.close("all")


def clean_z(z, c_dims):
    z_new = z.clone()
    for k in range(c_dims):
        z_new[0, k] = 0
    return z_new


# notebook utils
def get_colors_and_legend(c):
    cmap = plt.get_cmap(c)
    N = 20  # cmap.N
    cmap_a = cmap(np.arange(N))
    cmap_a[:, -1] = [0.3] * N
    cmap_a = np.vstack((cmap_a, [[0, 0, 0, 1]]*N))
    colors = {i: cmap_a[i] for i in range(0, N)}
    return colors, [matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=i, markerfacecolor=colors[i], markersize=10) for i in range(10)]


def produce_embeddings(dl, model, device, num_batches):
    xs, zs, ys, x_feats = [], [], [], []
    for bi, (x, y, _) in enumerate(dl.get_batch(device)):
        if bi >= num_batches:
            break
        x_feat, y_logits, z, z_mu_lvar, _ = model(x)
        xs.extend(x.detach().to("cpu"))
        z = model.encoding_head.construct_z(z_mu_lvar, add_noise=False)
        zs.extend(z.detach().to("cpu"))
        ys.extend(y.detach().to("cpu"))
        x_feats.extend(torch.reshape(x_feat, (x_feat.shape[0], -1)).detach().to("cpu"))
    xs, zs, ys, x_feats = torch.stack(xs, dim=0), torch.stack(zs), torch.stack(ys), torch.stack(x_feats)
    return xs, zs, ys, x_feats


def plots_embeddings(axes, ys, x_feats, zs, ld1, ld2):
    for ax in axes:
        ax.cla()
        ax.set_ylabel(f'Latent vector dim {ld1}')
        ax.set_xlabel(f'Latent vector dim {ld2}')
        colors, legend_handles = get_colors_and_legend('tab10')
        ax.legend(handles=legend_handles, loc='lower right', prop={'size': 6})
    axes[0].scatter(x_feats[:, ld1], x_feats[:, ld2], s=8, c=[colors[yy] for yy in ys.tolist()], picker=True)
    axes[1].scatter(zs[:, ld1], zs[:, ld2], s=8, c=[colors[yy] for yy in ys.tolist()], picker=True)


def interpolate_embeddings(endpts, axes, model, device, ld1, ld2, ninterps, norm_func):
    if len(endpts) < 2:
        return []
    x_feat_1, x_feat_2 = endpts[0]["x_feat"].to(device), endpts[1]["x_feat"].to(device)
    x_feats = [x_feat_1 + (x_feat_2 - x_feat_1)*t for t in np.linspace(0, 1, ninterps)]
    z_1, z_2 = endpts[0]["z"].to(device), endpts[1]["z"].to(device)
    zs = [z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, ninterps)]

    x_feats_zs = []
    for x_feat in x_feats:
        x_feat_z, _ = model.encoding_head(x_feat.unsqueeze(0), add_noise=False)
        x_feats_zs.append(x_feat_z.squeeze(0))
    x_feats_zs += zs
    interps = norm_func(model.decoder(torch.stack(x_feats_zs)).to('cpu').detach())

    axes[1].set_title(f'Interpolating from class {endpts[0]["y"]} to {endpts[1]["y"]}', y=-0.25)
    axes[0].imshow(make_grid(norm_func(endpts[0]["x"]), nrow=1).permute(1, 2, 0))
    axes[2].imshow(make_grid(norm_func(endpts[1]["x"]), nrow=1).permute(1, 2, 0))
    axes[1].imshow(make_grid(interps, nrow=ninterps).permute(1, 2, 0))
    axes[0].axis("on"); axes[1].axis("off"); axes[2].axis("on")  # noqa: E702
    return [torch.stack(x_feats, dim=0).detach().cpu().numpy(), torch.stack(zs, dim=0).detach().cpu().numpy()]
