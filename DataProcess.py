import os
import time
import pickle

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
import cx_Oracle as db

n_jbbm = 1332
n_jbbm_categ = 612
n_drug = 1125
n_drug_categ = 136


def dump_pkl(data, filename):
    output = open("./Med2Vec_data/" + filename, 'wb')
    pickle.dump(data, output, -1)
    output.close()


# 连接数据库
def connect():
    con = db.connect('MH3', '123456', '127.0.0.1:1521/ORCL')
    return con


# 执行select语句
def getSQL(sql, cursor):
    cursor.execute(sql)
    result = cursor.fetchall()
    content = []
    for row in result:
        if len(row) == 1:
            content.append(row[0])
        else:
            content.append(list(row))
    return content


# 获取数据库的部分内容
def create_dataset(cursor):
    sql = "SELECT DATA_ANALYSIS_JBBM_20170407.GRBH,DATA_ANALYSIS_JBBM_20170407.XH_INDEX,JBBM_INDEX," \
          "JBMC_CATEG_INDEX,DRUG_INDEX,DRUG_CATEG_INDEX FROM DATA_ANALYSIS_DRUG_20170407," \
          "DATA_ANALYSIS_JBBM_20170407 WHERE DATA_ANALYSIS_JBBM_20170407.XH=DATA_ANALYSIS_DRUG_20170407.XH " \
          "ORDER BY DATA_ANALYSIS_JBBM_20170407.GRBH,DATA_ANALYSIS_JBBM_20170407.ZYRQ"
    visits_ordered = getSQL(sql, cursor)
    seqs = []
    labels = []
    previous_grbh = '000000000000000023'
    previous_xh_index = 1
    seq = []
    label = []
    for i in visits_ordered:
        grbh = i[0]
        xh_index = i[1]
        if grbh != previous_grbh:
            seqs.append(seq)
            labels.append(label)
            seq = []
            label = []
            seqs.append([-1])
            labels.append([-1])
            previous_grbh = grbh
            previous_xh_index = xh_index
        elif xh_index != previous_xh_index:
            seqs.append(seq)
            labels.append(label)
            seq = []
            label = []
            previous_xh_index = xh_index
        if i[2] not in seq:
            seq.append(i[2])
        if i[3] not in label:
            label.append(i[3])
        if i[4] + n_jbbm not in seq:
            seq.append(i[4] + n_jbbm)
        if i[5] + n_jbbm_categ not in label:
            label.append(i[5] + n_jbbm_categ)
    dump_pkl(seqs, 'seqs.pkl')
    dump_pkl(labels, 'labels.pkl')


def show_relations(cursor):
    sql = 'select jbmc,JBBM_INDEX from MH3.DATA_ANALYSIS_JBBM_20170407 GROUP BY JBMC,JBBM_INDEX order by JBBM_INDEX'
    jbbm_ordered = getSQL(sql, cursor)
    sql = 'SELECT drug,drug_index+1332 from MH3.DATA_ANALYSIS_DRUG_20170407 group by drug,drug_index'
    drug_ordered = getSQL(sql, cursor)
    code_dict = {}
    for i in jbbm_ordered:
        code_dict[i[1]] = i[0]
    for i in drug_ordered:
        code_dict[i[1]] = i[0]
    dump_pkl(code_dict, 'code_dict.pkl')


if __name__ == '__main__':
    start = time.clock()
    # 连接数据库
    con = connect()
    cursor = con.cursor()
    # create_dataset(cursor=cursor)
    cursor.close()
    con.close()
    print(time.clock() - start)