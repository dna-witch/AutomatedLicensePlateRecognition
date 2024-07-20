FROM python:3.10.13-slim
COPY . ./
RUN pip3 install -r requirements.txt
CMD ["service.py"]
EXPOSE 8000
ENTRYPOINT [ "python" ]