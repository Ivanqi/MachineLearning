import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
import graphviz
import os


os.environ["PATH"] += os.pathsep + '/usr/local/lib/python3.8/site-packages/graphviz'
# 数据加载
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

# 数据探索
# 使用 info() 了解数据表的基本情况：行数、列数、每列的数据类型、数据完整度
print(train_data.info())
print('-' * 30)
# 使用 describe() 了解数据表的统计情况：总数、平均值、标准差、最小值、最大值等
print(train_data.describe())
print('-' * 30)
# 使用 describe(include=[‘O’]) 查看字符串类型（非数字）的整体情况
print(train_data.describe(include=['O']))
print('-' * 30)
# 使用 head 查看前几行数据（默认是前 5 行）
print(train_data.head())
print('-' * 30)
# 使用 tail 查看后几行数据（默认是最后 5 行）
print(train_data.tail())

# 数据清洗
# 使用平均年龄来填充年龄中的nan值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的nan值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

test_features = dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)

#  K 折交叉验证统计决策树准确率
print(u'cross_val_score 准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))
# 决策树可视化
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.view()
