from feast import FeatureView, Entity, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file_source import FileSource
import pandas as pd

iris_source = FileSource(
    path="/home/apoorvag_2017_gmail_com/ml-project/data/iris.parquet",
    event_timestamp_column="timestamp"
)
from feast import Entity, ValueType

species = Entity(name="species", value_type=ValueType.INT64)

iris_view = FeatureView(
    name="iris_view",
    entities=[species],
    ttl=None,
    schema=[
        Field(name="sepal length (cm)", dtype=Float32),
        Field(name="sepal width (cm)", dtype=Float32),
        Field(name="petal length (cm)", dtype=Float32),
        Field(name="petal width (cm)", dtype=Float32),
    ],
    source=iris_source
)
