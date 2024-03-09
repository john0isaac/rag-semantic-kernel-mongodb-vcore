#!/bin/bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m hypercorn app:app -b '0.0.0.0:50505'