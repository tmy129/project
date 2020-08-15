# project

## MultiCore_LDA.py

Run cmd: python3 MultiCore_LDA.py

1. Read text files
2. Build several LDA models
3. Compare model performance base on coherence values and output value chart
4. Select and save the best model (model.pkl)

## Run.py

1. A Flask API that gets best model coherence value
2. Include api logging

## unit_test_api.py

1. Performe three unit tests - model, api, and logging
- model unit test: check if gets return value
- api unit test: check if gets return value
- logging unit test: check app starts running
