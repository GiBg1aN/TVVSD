import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd


def plot_data(exp_data, column_name, axs, fmt='-'):
    """
    Plot 'exp_data' using an 'Errorplot'

    Args:
        exp_data: experiment data to plot
        column_name: the type of experiment to plot
        axs: matplotlib figure axes

    Returns:
        A tuple containing output of the plot for motion and
        non-motion verbs
    """
    # FILTERING
    filtered_data = exp_data[exp_data['representation_type'] == column_name]
    motion_rows = filtered_data[filtered_data['verb_type'] == 'motions']
    non_motion_rows = filtered_data[filtered_data['verb_type'] == 'non_motions']

    # STATISTICS
    motion_x_data = motion_rows['labels_per_class'].unique()
    motion_y_data = motion_rows.groupby('labels_per_class').mean().to_numpy()
    motion_y_err = motion_rows.groupby('labels_per_class').std().to_numpy()

    non_motion_x_data = non_motion_rows['labels_per_class'].unique()
    non_motion_y_data = non_motion_rows.groupby('labels_per_class').mean().to_numpy()
    non_motion_y_err = non_motion_rows.groupby('labels_per_class').std().to_numpy()


    axs[0].xaxis.set_ticks(np.arange(motion_x_data[0], motion_x_data[-1], 1))
    axs[1].xaxis.set_ticks(np.arange(motion_x_data[0], motion_x_data[-1], 1))
    axs[0].yaxis.set_ticks(np.arange(0.65, 1.0, 0.05))
    axs[1].yaxis.set_ticks(np.arange(0.65, 1.0, 0.05))
    axs[0].set_ylim(0.6, 1)
    axs[1].set_ylim(0.6, 1)
    motion_plot = axs[0].errorbar(motion_x_data, motion_y_data, yerr=motion_y_err, fmt=fmt)
    non_motion_plot = axs[1].errorbar(non_motion_x_data, non_motion_y_data, yerr=non_motion_y_err, fmt=fmt)

    return motion_plot, non_motion_plot


def main():
    """ Plot Semi-supervised experiment results """
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rc('xtick', labelsize=20)     
    matplotlib.rc('ytick', labelsize=20)

    columns = ['labels_per_class', 'verb_type', 'representation_type', 'accuracy']
    exp_data = pd.read_csv('generated/experiments.csv', names=columns)

    plt.style.use('seaborn-notebook')
    text_fig, text_axs = plt.subplots(ncols=2)
    cnn_fig, cnn_axs = plt.subplots(ncols=2)
    concat_fig, concat_axs = plt.subplots(ncols=2)

    # Uniform initialisation
    cap_motion_plot, _ = plot_data(exp_data, 'e_caption', text_axs)
    obj_motion_plot, _ = plot_data(exp_data, 'e_object', text_axs)
    text_motion_plot, _ = plot_data(exp_data, 'e_combined', text_axs)

    cnn_motion_plot, _ = plot_data(exp_data, 'e_image', cnn_axs)

    capcat_motion_plot, _ = plot_data(exp_data, 'concat_image_caption', concat_axs)
    objcat_motion_plot, _ = plot_data(exp_data, 'concat_image_object', concat_axs)
    textcat_motion_plot, _ = plot_data(exp_data, 'concat_image_text', concat_axs)

    text_plots = (cap_motion_plot, obj_motion_plot, text_motion_plot)
    cnn_plots = (cnn_motion_plot)
    concat_plots = (capcat_motion_plot, objcat_motion_plot, textcat_motion_plot)

    text_legend_labels = ('Captions', 'Objects annotations', 'Captions+Objects')
    cnn_legend_labels = ('CNN')
    concat_legend_labels = ('CNN+Captions', 'CNN+Objects', 'CNN+Captions+Objects')

    # Text plot settings
    text_axs[0].set(title='Motion verbs', xlabel='#labels', ylabel='Accuracy')
    text_axs[1].set(title='Non-motion verbs', xlabel='#labels', ylabel='Accuracy')
    text_fig.suptitle('Semi-supervised GTG (text data)')
    text_fig.legend(text_plots, text_legend_labels, loc='lower center', ncol=6)

    # CNN plot settings
    cnn_axs[0].set(title='Motion verbs', xlabel='#labels', ylabel='Accuracy')
    cnn_axs[1].set(title='Non-motion verbs', xlabel='#labels', ylabel='Accuracy')
    cnn_fig.suptitle('Semi-supervised GTG (visual data)')
    # cnn_fig.legend(cnn_plots, cnn_legend_labels, loc='lower center')

    # Concat plot settings
    concat_axs[0].set(title='Motion verbs', xlabel='#labels', ylabel='Accuracy')
    concat_axs[1].set(title='Non-motion verbs', xlabel='#labels', ylabel='Accuracy')
    concat_fig.suptitle('Semi-supervised GTG (Visual and textual concatenation)')
    concat_fig.legend(concat_plots, concat_legend_labels, loc='lower center', ncol=3)

    plt.show()


if __name__ == '__main__':
    main()
