FROM registry-vpc.cn-shenzhen.aliyuncs.com/spark-base/anaconda3:5.2.0

WORKDIR /app

COPY requirements.txt requirements.txt

RUN set -x \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["tail","-f","/dev/stderr"]
