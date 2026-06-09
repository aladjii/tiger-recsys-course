class Constants:
    HOUR_SECONDS = 60 * 60
    DAY_SECONDS = 24 * HOUR_SECONDS

    GAP_SIZE = HOUR_SECONDS // 2
    VAL_SIZE = 1 * DAY_SECONDS
    TEST_SIZE = 1 * DAY_SECONDS

    LAST_TIMESTAMP = 26000000
    TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE

    TRACK_LISTEN_THRESHOLD = 50

    NUM_RANKED_ITEMS = 100

    METRICS = [
        "ndcg@10",
        "ndcg@50",
        "ndcg@100",
        "dcg@10",
        "dcg@50",
        "dcg@100",
        "recall@10",
        "recall@50",
        "recall@100",
        "coverage@10",
        "coverage@50",
        "coverage@100",
    ]
