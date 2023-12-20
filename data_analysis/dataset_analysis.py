from matplotlib import pyplot as plt
from data_analysis.dataset import load_dataset


def create_listeners():
    data = load_dataset()

    classical_listeners = data[data['Fav genre'] == 'Classical']
    # print(classical_listeners)

    edm_listeners = data[data['Fav genre'] == 'EDM']
    # print(edm_listeners)

    metal_listeners = data[data['Fav genre'] == 'Metal']
    # print(metal_listeners)

    pop_listeners = data[data['Fav genre'] == 'Pop']
    # print(pop_listeners)

    return classical_listeners, edm_listeners, metal_listeners, pop_listeners


""" 
Frequency of listening to music genres
    Scale of disorders: 1-10
    Frequency of disorders: Never, Rarely, Sometimes, Very Frequent
"""


def generate_frequency():
    classical_listeners, edm_listeners, metal_listeners, pop_listeners = create_listeners()

    print("--Classical--")
    frequency(classical_listeners, "EDM")
    frequency(classical_listeners, "Metal")
    frequency(classical_listeners, "Pop")
    print("--EDM--")
    frequency(edm_listeners, "Classical")
    frequency(edm_listeners, "Metal")
    frequency(edm_listeners, "Pop")
    print("--Metal--")
    frequency(metal_listeners, "Classical")
    frequency(metal_listeners, "EDM")
    frequency(metal_listeners, "Pop")
    print("--Pop--")
    frequency(pop_listeners, "Classical")
    frequency(pop_listeners, "EDM")
    frequency(pop_listeners, "Metal")


def frequency(set, title):
    frequency_counts = set[f'Frequency [{title}]'].value_counts()
    most_frequent_frequency = frequency_counts.idxmax()
    print(f"The most frequent value in the 'Frequency [{title}]' column:",
          most_frequent_frequency)


"""
Mean intensity of mental disorders for selected music genres
"""


def generate_mean():
    anxiety_means, depression_means, insomnia_means = mean()

    mean_visualization(anxiety_means, "Anxiety")
    mean_visualization(depression_means, "Depression")
    mean_visualization(insomnia_means, "Insomnia")


def mean():
    classical_listeners, edm_listeners, metal_listeners, pop_listeners = create_listeners()

    classical_anxiety_average = classical_listeners['Anxiety'].mean()
    edm_anxiety_average = edm_listeners['Anxiety'].mean()
    metal_anxiety_average = metal_listeners['Anxiety'].mean()
    pop_anxiety_average = pop_listeners['Anxiety'].mean()

    classical_depression_average = classical_listeners['Depression'].mean()
    edm_depression_average = edm_listeners['Depression'].mean()
    metal_depression_average = metal_listeners['Depression'].mean()
    pop_depression_average = pop_listeners['Depression'].mean()

    classical_insomnia_average = classical_listeners['Insomnia'].mean()
    edm_insomnia_average = edm_listeners['Insomnia'].mean()
    metal_insomnia_average = metal_listeners['Insomnia'].mean()
    pop_insomnia_average = pop_listeners['Insomnia'].mean()

    anxiety_means = [classical_anxiety_average, edm_anxiety_average, metal_anxiety_average, pop_anxiety_average]

    depression_means = [classical_depression_average, edm_depression_average, metal_depression_average,
                        pop_depression_average]

    insomnia_means = [classical_insomnia_average, edm_insomnia_average, metal_insomnia_average, pop_insomnia_average]

    return anxiety_means, depression_means, insomnia_means


def mean_visualization(means, disorder):
    colors = ['#5DADE2', '#48C9B0', '#34495E', '#E74C3C']
    genres = ['Classical', 'EDM', 'Metal', 'Pop']

    plt.bar(genres, means, color=colors)
    plt.title(f'Average {disorder} by Music Genre')
    plt.xlabel('Music Genre')
    plt.ylabel(disorder)
    plt.savefig(f"plots/means/{disorder}_mean.png")
    plt.show()
