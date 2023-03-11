"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["x_train", "x_val", "y_train", "y_val", "params:model_params"],
                outputs="ag_news_tf_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["ag_news_tf_model", "x_test", "y_test"],
                outputs="confusion_matrix",
                name="evaluate_model_node",
            ),
        ]
    )
