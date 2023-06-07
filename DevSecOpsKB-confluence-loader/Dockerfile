FROM python:3.11.0

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
ENV PATH="/usr/bin/tesseract:${PATH}"
EXPOSE 7860
CMD ["python", "kb.py"]
