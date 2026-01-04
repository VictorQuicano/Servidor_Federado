#!/bin/bash

# Activar entorno si es necesario o correr directamente con uvicorn
uvicorn main:app --host 0.0.0.0 --port 8081 --reload
