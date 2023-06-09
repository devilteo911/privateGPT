from datetime import datetime


def pick_logs_filename(**kwargs):
    """
    Checks whether it is an huggingface model name or a path and assign the history chat name
    as a consequence
    """
    # TODO: handle local models in different locations

    model_path = kwargs["model_path"]
    emb_path = kwargs["embeddings_model_name"]
    logs_filename = ""

    if len(emb_path.split("/")) == 2:
        # usually huggingface models are 'username/model_name'
        logs_filename = "{}_{}_{}_{}_{}.txt".format(
            model_path.split("/")[-1],
            emb_path.split("/")[-1],
            kwargs["model_n_ctx"],
            kwargs["target_source_chunks"],
            datetime.now(),
        )
    else:
        logs_filename = "{}_{}_{}_{}_{}.txt".format(
            model_path.split("/")[-1],
            (emb_path.split("/")[2]).split("--")[1],
            kwargs["model_n_ctx"],
            kwargs["target_source_chunks"],
            datetime.now(),
        )
    return logs_filename
