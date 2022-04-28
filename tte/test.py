from sksurv.datasets import load_gbsg2
from lifelines.datasets import load_rossi, load_kidney_transplant


df = load_kidney_transplant()

print(df.isna().sum())