import re
import jieba

# 定义工具函数，取出一个词语中每个字的标记
def get_tag(word):
    tag = []

    if len(word) == 1:
        tag = ['S']
    elif len(word) == 2:
        tag = ['B', 'E']
    else:
        num = len(word) - 2
        tag.append('B')
        tag.extend(['M'] * num)
        tag.append('E')
    
    return tag


# 初始化矩阵
def mats_setup(STATES):
    trans_mat = {}
    emit_mat = {}
    init_vec = {}
    state_count = {}

    for state in STATES:
        trans_mat[state] = {}

        for target in STATES:
            trans_mat[state][target] = 0.0
        
        emit_mat[state] = {}
        init_vec[state] = 0

        state_count[state] = 0
    
    return trans_mat, emit_mat, init_vec, state_count


# 定义训练程序，得到初始概率向量，状态转移矩阵和发射矩阵
def get_Mats_fenci(filename):
    word_file = open(filename, 'r', encoding='utf-8').readlines()

    seg_stop_words = {" ","，","。","“","”",'“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’", 
                "──",",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", 
                "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n","\t"}
    
    trans_mat, emit_mat, init_vec, state_count = mats_setup(['B', 'M', 'E', 'S'])

    for line in word_file:
        line = line.strip()
        if not line:
            continue
            
        # 获取观测序列
        observes = []
        for i in range(len(line)):
            if line[i] not in seg_stop_words:
                observes.append(line[i])
            
        words = line.split(" ")

        # 获取实际状态序列
        states = []
        for word in words:
            if word not in seg_stop_words:
                states.extend(get_tag(word))
        
        # 计数，记频率
        if (len(observes) >= len(states)):
            for i in range(len(states)):
                if i == 0:
                    init_vec[states[0]] += 1
                    state_count[states[0]] += 1
                else:
                    trans_mat[states[i - 1]][states[i]] += 1
                    state_count[states[i]] += 1
                
                if observes[i] not in emit_mat[states[i]]:
                    emit_mat[states[i]][observes[i]] = 1
                else:
                    emit_mat[states[i]][observes[i]] += 1
                
        else:
            pass
    
    return init_vec, trans_mat, emit_mat, state_count

# 将频数转换成频率
def get_Prob(init_vec, trans_mat, emit_mat, state_count):
    init_vec1 = {}
    trans_mat1 = {}
    emit_mat1 = {}
    asum = sum(init_vec.values())

    for key1 in trans_mat:
        trans_mat1[key1] = {}

        for key2 in trans_mat[key1]:
            if state_count[key1] != 0:
                trans_mat1[key1][key2] = float(trans_mat[key1][key2]) / state_count[key1]
            else:
                trans_mat1[key1][key2] = float(trans_mat[key1][key2]) / default
    
    for key1 in emit_mat:
        emit_mat1[key1] = {}
        for key2 in emit_mat[key1]:
            if state_count[key1] != 0:
                emit_mat1[key1][key2] = float(emit_mat[key1][key2]) / state_count[key1]
            else:
                emit_mat1[key1][key2] = float(emit_mat[key1][key2]) / default
    
    return init_vec1, trans_mat1, emit_mat1


# 维特比算法，做预测
def viterbi(sequence, EPS, init_vec, trans_mat, emit_mat, STATES):
    tab = [{}]
    path = {}

    for state in STATES:
        tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0], EPS)
        path[state] = [state]
    
    # 创建动态搜索表
    for t in range(1, len(sequence)):
        tab.append({})
        new_path = {}
        
        for state1 in STATES:
            items = []
            for state2 in STATES:
                if tab[t - 1][state2] == 0:
                    continue
                
                prob = tab[t - 1][state2] * trans_mat[state2].get(state1, EPS) * emit_mat[state1].get(sequence[t], EPS)
                items.append((prob, state2))
            
            best = max(items)
            tab[t][state1] = best[0]
            new_path[state1] = path[best[1]] + [state1]
        
        path = new_path
    
    # 搜索最优路径
    prob, state = max([(tab[len(sequence) - 1][state], state) for state in STATES])
    return prob, state, path

# 由状态转为分词后的句子的程序
def stateToFenci(state, sequence):
    fenci = ''
    for i in range(len(path[state])):
        j = path[state][i]

        if j == 'B':
            fenci = fenci + sequence[i]
        else:
            if j == 'M':
                fenci = fenci + sequence[i]
            else:
                fenci = fenci + sequence[i] + ' '
    
    return fenci

# 一次分词试验
sequence = ' 没父亲的宋志高同学从小就机灵'
EPS = 0.00001
training_file = './data/pku_training.utf8'

init_vec, trans_mat, emit_mat, state_count = get_Mats_fenci(training_file)
# init_vec1, trans_mat1, emit_mat1 = get_Prob(init_vec, trans_mat, emit_mat,state_count)
print(init_vec)
print(trans_mat)
# prob, state, path = viterbi(sequence, EPS, init_vec1, trans_mat1, emit_mat1, ['B', 'M', 'E', 'S'])

# print(stateToFenci(state,sequence))
# print(path[state])
