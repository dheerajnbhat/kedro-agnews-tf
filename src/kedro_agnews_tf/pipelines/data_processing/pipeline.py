"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from .nodes import preprocess_data, train_val_split
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["ag_news_train", "ag_news_test"],
                outputs=["x", "x_test", "y", "y_test"],
                name="preprocess_data_node"
            ),
            node(
                func=train_val_split,
                inputs=["x", "y", "params:data_params"],
                outputs=["x_train", "x_val", "y_train", "y_val"],
                name="train_val_split_node",
            ),
        ]
    )
