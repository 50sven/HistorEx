import pandas as pd
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity

NUM_DOCS = 308


def get_most_similar_tokens(input_token, cos_sim_matrix, kind="tokens", num=10, places=None, persons=None):
    """Retrieve most similar tokens, persons, places and/or documents

    Args:
        input_token (string):  input of interest (token of any kind for which an embedding exists)
        cos_sim_matrix (DataFrame): cosine similarity matrix
        kind (string): type of entity which are of interest
                       (one of {"tokens, "docs", "persons", "places"})
        num (int): number of entities returned
        persons (list): list of persons for which an embeddings exists
                        (mandatory if kind is "persons")
        places (list): list of places for which an embeddings exists
                       (mandatory if kind is "places")

    Returns:
        output_tokens (list): list of most similar tokens (strings)
        similarities (list): list of cosine similarities according to the output tokens
    """
    assert input_token in cos_sim_matrix.columns.tolist(), "The input token does not exists."

    sim_vec = cos_sim_matrix.loc[input_token]

    if kind == "places":
        sim_vec = sim_vec[places]
    if kind == "persons":
        sim_vec = sim_vec[persons]
    if kind == "docs":
        docs = cos_sim_matrix.columns[:NUM_DOCS]
        sim_vec = sim_vec[docs]
    if kind == "tokens":
        docs = cos_sim_matrix.columns[:NUM_DOCS]
        sim_vec = sim_vec.drop(docs)

    tokens_sorted = sim_vec.sort_values(ascending=True)
    if input_token == tokens_sorted.index[-1]:
        tokens_sorted = tokens_sorted[:-1]
    output_tokens = tokens_sorted.index.tolist()[-num:]
    if (kind == "persons" or kind == "places"):
        output_tokens = [" ".join(re.findall("[A-Z]+[a-z]*", p)) for p in output_tokens]
    similarities = tokens_sorted.tolist()

    return output_tokens, similarities


def get_cosine_similarity(embeddings, mapping):
    """Retrieve cosine similarity matrix
    """

    cos_sim_matrix = cosine_similarity(embeddings)
    cos_sim_matrix_df = pd.DataFrame(cos_sim_matrix)
    cos_sim_matrix_df.columns = mapping.keys()
    cos_sim_matrix_df.index = mapping.keys()

    return cos_sim_matrix_df


################################################################################################################


if __name__ == "__main__":

    # embeddings = pickle.load(open('./assets/data_embeddings.pkl', 'rb'))
    # id_mapping = pickle.load(open("./assets/data_id_mapping.pkl", "rb"))
    # cos_sim_matrix = get_cosine_similarity(embeddings, id_mapping)

    remaining_persons = pickle.load(open('./assets/data_remaining_persons.pkl', 'rb'))
    remaining_places = pickle.load(open('./assets/data_remaining_places.pkl', 'rb'))
    cos_sim_matrix = pd.read_pickle("./assets/data_cosine_similarity_matrix.pkl")

    tokens, similarities = get_most_similar_tokens("Abraham", cos_sim_matrix, kind="persons",
                                                   num=10, places=None, persons=remaining_persons)
