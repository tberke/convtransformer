class Config:
    """Default preprocessing, model and training parameters."""

    # Folder where the trained models will be saved
    dir_saves = "save/"

    # Folder where the data is stored
    dir_data = "data/"

    # Names of the train, validation and test files
    file_train_src = "train.src"
    file_train_tgt = "train.tgt"
    file_val_src = "validation.src"
    file_val_tgt = "validation.tgt"
    file_test_src = "test.src"
    file_test_tgt = "test.tgt"

    # Name of the file containing the character vocabulary
    file_alph = "alphabet"

    # Special characters
    chars_special = {
        "padding": "\N{WHITE SQUARE}",
        "start": "\N{RIGHTWARDS ARROW}",
        "end": "\N{LEFTWARDS ARROW}",
        "unknown": "\N{BLACK SQUARE}",
    }

    # Names of the raw source and target files
    files_raw = {"Europarl.de-fr.fr": "Europarl.de-fr.de"}

    # URLs for the zip-files containing the raw source and target files
    urls_raw = {
        "Europarl.de-fr.fr": "http://opus.nlpl.eu/download.php"
        "?f=Europarl/v8/moses/de-fr.txt.zip",
        "Europarl.de-fr.de": "http://opus.nlpl.eu/download.php"
        "?f=Europarl/v8/moses/de-fr.txt.zip",
    }

    # Preprocessing parameters
    min_char_freq = 1 / 100000
    size_data_train = -1
    size_data_val = 1500
    size_data_test = 1500

    # Model parameters
    max_len = 300
    sz_kernels = [3, 5, 7]
    sz_kernel_final = 3
    sz_emb = 512
    num_lay = 6
    dim_ff = 2048
    nhead = 8

    # Regularization parameters
    dropout = 0.1
    label_smoothing = 0.1

    # Optimizer parameters
    adam_betas = (0.9, 0.98)
    adam_eps = 1e-09

    # Scheduler parameters
    base_lr = 0.0001
    warmup_init_lr = 1e-07
    warmup_steps = 4000

    # Maximal batch size (measured in characters)
    sz_batch = 4096

    # Minimal bucket size (measured in sentence pairs)
    sz_bucket = 20000

    # Number of epochs to train
    nr_epochs = 10

    # Print statistics on screen every 'log_interval' seconds
    log_interval = 15 * 60

    # Save the model every 'save_interval' seconds
    save_interval = 120 * 60
