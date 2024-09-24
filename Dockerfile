# 
FROM python:3.8.6-slim

#
WORKDIR /B17PGH
#
COPY ./runtime.txt /B17PGH/runtime.txt

# 
COPY ./requirements.txt /B17PGH/requirements.txt

COPY ./main.py /B17PGH/main.py
COPY ./templates /B17PGH/templates
# 
RUN pip install --no-cache-dir --upgrade -r /B17PGH/requirements.txt
#
EXPOSE 8000/tcp

EXPOSE 8000/udp
# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
