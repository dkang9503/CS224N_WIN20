#!/bin/bash
python backtranslation.py --input=train.csv --lang=es
python backtranslation.py --input=train.csv --lang=ja
python backtranslation.py --input=train.csv --lang=de