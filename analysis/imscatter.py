import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Function for weight / height scatter plot with imported images

def main():

    _, ax = plt.subplots()

    for x, y, image_path in zip(df['weight_kgs'], df['height_m'], df['sprite']):
        imscatter(x, y, image_path, zoom=0.7, ax=ax)
        ax.scatter(x, y)

    plt.title('Pokémon Sprites by Weight and Height', fontsize=20)
    plt.xlabel("Weight", fontsize=20)
    plt.ylabel("Height", fontsize=20)

    # Saves file in directory

    plt.savefig('poke_weight.png')


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except (TypeError, AttributeError):
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


if __name__ == "__main__":
    plt.style.use(style='ggplot')
    plt.rcParams['figure.figsize'] = (22, 14)
    df = pd.read_csv(
        'https://raw.githubusercontent.com/gabrielvcbessa/pokemon/master/pokemon_complete.csv')
    main()
    plt.show()
