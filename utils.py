from matplotlib import pyplot as plt
import math

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

def show_image_with_title_on_subplot(image, title, subplot, titlesize=18):
    plt.subplot(*subplot)
    plt.imshow(image)
    plt.gcf().set_facecolor('#9370DB')
    if len(title) > 0:
        plt.title(title, fontsize=int(20), color='#FFD700', fontdict={'verticalalignment': 'center'},
                  pad=int(titlesize / 1.5))
    return (subplot[0], subplot[1], subplot[2] + 1)

def show_image_batch_with_predictions(images, predictions):
    images = [image.numpy_view() for image in images]

    num_images = len(images)
    rows = int(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    # Size and spacing
    FIGSIZE = 4.0
    SPACING = 0.1

    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))
    # Display.
    for i, (image, prediction) in enumerate(zip(images[:rows * cols], predictions[:rows * cols])):
        dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 20 + 3
        subplot = show_image_with_title_on_subplot(image, prediction, subplot, titlesize = dynamic_titlesize)
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()
