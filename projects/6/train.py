
import os, sys

from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from joblib import dump


def main():
            path_in = sys.argv[2] 
            path_out = sys.argv[4] 
            dataset = pd.read_json(path_in, lines=True)

            all_features = []
            for r in dataset['features']:
                l = []
                if len(r['values']) == 100:
                    all_features.append(r['values'])
                else:
                    for i in range(100):
                        if i in r['indices']:
                            l.append(r['values'][r['indices'].index(i)])
                        else:
                            l.append(0)
                    all_features.append(l)

            df = pd.DataFrame(all_features)

            target = dataset['label']

            #feats = list(features.columns)[1:]

            model = GradientBoostingClassifier()
            model.fit(df, target)

            dump(model, path_out)

if __name__ == "__main__":
    main()

