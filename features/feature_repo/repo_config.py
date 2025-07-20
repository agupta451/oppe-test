from feature_view import iris_view, species
from feast import FeatureStore

store = FeatureStore(repo_path=".")
store.apply([iris_view, species])
store.materialize_incremental(end_date=pd.to_datetime("2021-12-31"))
