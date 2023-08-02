import os
import torch
from flask import g
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer,util
import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug(os.getcwd())

# torch.cuda.is_available = lambda: False
model = SentenceTransformer('./sentence-transformers/all-MiniLM-L6-v2', device='cpu')
# model = SentenceTransformer('./sentence-transformers/all-MiniLM-L6-v2', device='cuda')
print(model.device)

def get_es():
    if 'es' not in g:
        g.es = Elasticsearch('https://172.16.10.6:9200', ca_certs="./http_ca.crt", basic_auth=('elastic', "test001"))
    return g.es


def close_es(e=None):
    es = g.pop('es', None)
    if es is not None:
        es.close()


def init_app(app):
    app.teardown_appcontext(close_es)


def query_vector(keyword):
    es = get_es()
    logger.info('info:', 1)
    query = model.encode([keyword], convert_to_tensor=True)
    query = util.normalize_embeddings(query)[0].tolist()
    # print(query.shape)
    # print(es.search(index='ws'))
    res = es.knn_search(index='ws', source=["paragraph", ], knn={"field": "vector",  "query_vector": query, "k": 10, "num_candidates": 4000})
    # print(res['hits']['hits'])
    ps = [source['_source']['paragraph'] for source in res['hits']['hits']]
    print(len(ps))
    ps = random.sample(ps, 3)
    return ps


def query_keyword(keyword):
    es = get_es()
    query = {
        'query': {
            # 'match': {
            #     'paragraph': keyword.split()
            # }
            'terms': {
                'paragraph': keyword.split()
            }
        }
    }
    response = es.search(index='ws',source=["paragraph", ], body=query)
    # print(response)
    try:
        ps = [source['_source']['paragraph'] for source in response['hits']['hits']][:3]
    except Exception as e:
        ps = []
    print(len(ps))
    return ps


def get_db_local():
    import pymysql
    HOST = '172.16.10.6'
    PORT = 3306
    USER = 'product'
    PASSWD = 'pl,okm098'
    db = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWD)
    return db


doc = {
    'mappings': {
        'properties': {
            'vector': {
                'type': 'dense_vector',
                'dims': 384,
                'index': True,
                # 'similarity': "cosine",
                'similarity': "dot_product"
                # "similarity": "l2_norm"
            },
            'paragraph': {
                'type': 'text',
                'fields': {
                    'keyword': {
                        'type': 'keyword',
                        'index': True
                    }
                }
            }

        }
    }
}


def update_index_es():
    es = get_es()
    es.indices.delete(index='ws')
    response = es.indices.create(index='ws', body=doc)
    # print(response)
    db = get_db_local()
    cursor = db.cursor()

    search_sql = f'SELECT  `paragraphs` from `ws_data_collect`.`ws_files`;'
    cursor.execute(search_sql)
    paragraphs = cursor.fetchall()

    paragraph = ['\n'.join(eval(p[0])[-2:]) for p in paragraphs]
    paragraphs = ['\n'.join(eval(p[0])) for p in paragraphs]
    vector = model.encode(paragraph, convert_to_tensor=True)
    vector = util.normalize_embeddings(vector)

    # ws_embedding = util.normalize_embeddings(vector)
    # data_list = [{'_op_type': 'index', '_index': 'ws', 'doc':{'vector': v.tolist(), 'paragraph': p}} for v, p in zip(vector, paragraphs)]
    # response = helpers.bulk(es, data_list)
    # print(response)
    for p, v in zip(paragraphs, vector):
        _ = {
            'paragraph': p,
            'vector': v.tolist()
        }
        es.index(index='ws', body=_)
