import sys  # 导入 sys，用于在发生错误时终止程序
import numpy as np  # 导入 numpy，用于数值计算
import pandas as pd  # 导入 pandas，用于表格数据读取与处理
from pathlib import Path  # 导入 Path，方便处理文件路径
from scipy.stats import mannwhitneyu, spearmanr  # 导入 Mann-Whitney U 检验和 Spearman 相关系数
from sklearn.impute import SimpleImputer  # 导入缺失值填补器
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder  # 导入分箱器和标签编码器
from sklearn.metrics import mutual_info_score  # 导入互信息计算函数

INPUT_CSV = Path("Total.csv")  # 输入文件路径，这里使用你的 Total.csv
OUTPUT_DIR = Path("feature_selection_results")  # 输出文件夹路径
OUTPUT_DIR.mkdir(exist_ok=True)  # 如果输出文件夹不存在则自动创建

LABEL_COL_INDEX = 1  # 标签列索引，按 Python 从 0 开始计数，这里第 2 列为标签
FEATURE_START_COL_INDEX = 2  # 特征起始列索引，这里第 3 列开始为特征
ENCODING = "utf-8-sig"  # CSV 文件编码格式，适合中文环境导出
MWU_P_THRESHOLD = 0.05  # Mann-Whitney U 检验的显著性阈值
SPEARMAN_THRESHOLD = 0.90  # Spearman 去冗余阈值，若两特征相关性绝对值大于该值则删除其中一个
TOP_K = 50  # mRMR 最终保留的特征数量
MRMR_CRITERION = "MID"  # mRMR 评价准则，可选 "MID" 或 "MIQ"
N_BINS = 10  # mRMR 前离散化分箱数
RANDOM_STATE = 42  # 随机种子，保证结果可复现
EPS = 1e-12  # 防止 MIQ 分母为 0 的极小值

def discrete_mi(x_disc: np.ndarray, y_disc: np.ndarray) -> float:  # 定义离散变量之间的互信息计算函数
    return float(mutual_info_score(x_disc, y_disc))  # 使用 sklearn 的 mutual_info_score 计算互信息并返回浮点数

def load_and_prepare_data(path: Path):  # 定义读取并预处理数据的函数
    if not path.exists():  # 如果输入文件不存在
        sys.exit(f"[ERROR] File not found: {path}")  # 终止程序并报错

    df = pd.read_csv(path, encoding=ENCODING)  # 读取 CSV 文件
    if df.shape[1] <= FEATURE_START_COL_INDEX:  # 如果列数不足，说明至少要有 ID、标签和特征
        sys.exit("[ERROR] 数据列数不足，至少应包含：ID列 + 标签列 + 1个特征列。")  # 终止程序并提示

    df.iloc[:, LABEL_COL_INDEX] = pd.to_numeric(df.iloc[:, LABEL_COL_INDEX], errors="coerce")  # 将标签列转为数值，无法转换的设为 NaN
    X_all = df.iloc[:, FEATURE_START_COL_INDEX:].apply(pd.to_numeric, errors="coerce")  # 将所有特征列转为数值，无法转换的设为 NaN

    df.iloc[:, LABEL_COL_INDEX] = df.iloc[:, LABEL_COL_INDEX].fillna(df.iloc[:, LABEL_COL_INDEX].mean())  # 用标签列均值填补缺失值
    X_all = X_all.fillna(X_all.mean())  # 每个特征列用各自均值填补缺失值

    y = df.iloc[:, LABEL_COL_INDEX].astype(int)  # 将标签列转换为整数型，适用于二分类任务
    X = X_all.copy()  # 复制特征矩阵，避免后续修改原始对象

    print(f"样本量: {len(df)}，原始特征数: {X.shape[1]}")  # 打印样本量和原始特征数

    unique_y = sorted(y.unique().tolist())  # 获取标签中的唯一值并排序
    if len(unique_y) != 2:  # 如果不是二分类
        sys.exit(f"[ERROR] 当前标签列不是二分类，检测到类别为: {unique_y}，Mann-Whitney U 检验适用于二分类。")  # 终止程序并报错

    return df, X, y  # 返回原始数据、特征矩阵和标签向量

def mann_whitney_filter(X: pd.DataFrame, y: pd.Series):  # 定义 Mann-Whitney U 检验筛选函数
    p_values = {}  # 创建字典保存每个特征的 p 值

    class_values = sorted(y.unique())  # 获取两类标签值并排序
    class0 = class_values[0]  # 第一类标签
    class1 = class_values[1]  # 第二类标签

    for col in X.columns:  # 遍历每一个特征
        grp0 = X.loc[y == class0, col]  # 取出 class0 组该特征的所有样本值
        grp1 = X.loc[y == class1, col]  # 取出 class1 组该特征的所有样本值

        try:  # 尝试执行检验
            _, p = mannwhitneyu(grp1, grp0, alternative="two-sided")  # 进行双侧 Mann-Whitney U 检验
        except Exception:  # 如果该特征出现异常
            p = 1.0  # 将其 p 值设为 1，表示不显著

        p_values[col] = p  # 保存该特征对应的 p 值

    p_series = pd.Series(p_values).sort_values()  # 将 p 值字典转换为 Series 并按升序排序
    sig_series = p_series[p_series < MWU_P_THRESHOLD]  # 保留 p 小于阈值的显著特征
    selected_features = sig_series.index.tolist()  # 获取显著特征名称列表
    X_mwu = X[selected_features].copy()  # 根据显著特征提取新特征矩阵

    p_series.to_csv(OUTPUT_DIR / "01_MWU_p_values.csv", header=["p_value"], encoding=ENCODING)  # 保存全部特征的 p 值结果
    sig_series.to_csv(OUTPUT_DIR / "02_MWU_significant_features.csv", header=["p_value"], encoding=ENCODING)  # 保存显著特征结果

    print(f"Mann-Whitney U 检验后保留特征数: {X_mwu.shape[1]}")  # 打印 MWU 初筛后的特征数量

    return X_mwu, p_series, sig_series  # 返回筛选后的特征矩阵、全部 p 值和显著 p 值

def spearman_reduction(X: pd.DataFrame, p_series: pd.Series):  # 定义基于 Spearman 相关性的去冗余函数
    if X.shape[1] <= 1:  # 如果特征数小于等于 1，则无需去冗余
        print("Spearman 去冗余跳过：当前特征数 <= 1")  # 打印提示
        return X.copy(), [], pd.DataFrame()  # 直接返回原始特征、空删除记录和空相关矩阵

    corr_matrix = X.corr(method="spearman").abs()  # 计算 Spearman 绝对值相关矩阵
    cols = corr_matrix.columns.tolist()  # 获取特征名列表
    to_drop = set()  # 创建集合记录需要删除的特征
    drop_records = []  # 创建列表记录删除详情

    for i in range(len(cols)):  # 外层循环遍历每个特征
        for j in range(i + 1, len(cols)):  # 内层循环遍历当前特征之后的所有特征，避免重复比较
            col_i = cols[i]  # 第 i 个特征名
            col_j = cols[j]  # 第 j 个特征名
            corr_val = corr_matrix.loc[col_i, col_j]  # 获取两特征之间的绝对相关系数

            if corr_val > SPEARMAN_THRESHOLD:  # 若相关性超过阈值，则认为高度冗余
                p_i = p_series.get(col_i, 1.0)  # 获取第一个特征的 MWU p 值，若不存在则默认 1
                p_j = p_series.get(col_j, 1.0)  # 获取第二个特征的 MWU p 值，若不存在则默认 1

                if p_i <= p_j:  # 如果第一个特征 p 值更小或相等，说明区分能力更强
                    drop_feature = col_j  # 删除第二个特征
                    keep_feature = col_i  # 保留第一个特征
                else:  # 如果第二个特征 p 值更小
                    drop_feature = col_i  # 删除第一个特征
                    keep_feature = col_j  # 保留第二个特征

                if drop_feature not in to_drop:  # 如果该待删除特征尚未记录
                    to_drop.add(drop_feature)  # 将其加入删除集合
                    drop_records.append([keep_feature, drop_feature, corr_val, p_series.get(keep_feature, np.nan), p_series.get(drop_feature, np.nan)])  # 记录保留/删除详情

    X_reduced = X.drop(columns=list(to_drop), errors="ignore").copy()  # 删除冗余特征并生成去冗余后的特征矩阵

    drop_df = pd.DataFrame(drop_records, columns=["kept_feature", "dropped_feature", "abs_spearman_corr", "kept_feature_mwu_p", "dropped_feature_mwu_p"])  # 将删除记录整理成表格
    corr_matrix.to_csv(OUTPUT_DIR / "03_Spearman_correlation_matrix.csv", encoding=ENCODING)  # 保存 Spearman 相关矩阵
    drop_df.to_csv(OUTPUT_DIR / "04_Spearman_dropped_features.csv", index=False, encoding=ENCODING)  # 保存删除详情表
    X_reduced.to_csv(OUTPUT_DIR / "05_Features_after_Spearman.csv", index=False, encoding=ENCODING)  # 保存去冗余后的特征表

    print(f"Spearman 去冗余后保留特征数: {X_reduced.shape[1]}")  # 打印 Spearman 去冗余后的特征数量

    return X_reduced, drop_records, corr_matrix  # 返回去冗余后的特征矩阵、删除记录和相关矩阵

def mrmr_select(X: pd.DataFrame, y: pd.Series, top_k: int = TOP_K, criterion: str = MRMR_CRITERION):  # 定义 mRMR 特征选择函数
    if X.shape[1] == 0:  # 如果没有输入特征
        print("mRMR 跳过：输入特征数为 0")  # 打印提示
        return [], pd.DataFrame(), pd.DataFrame()  # 返回空结果

    feat_names = X.columns.tolist()  # 获取当前特征名列表

    y_enc = LabelEncoder().fit_transform(y.astype(str).values)  # 将标签编码为整数形式，适用于互信息计算

    imp = SimpleImputer(strategy="median")  # 创建中位数填补器
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=feat_names, index=X.index)  # 对输入特征再次用中位数填补缺失值

    actual_bins = min(N_BINS, max(2, int(X_imp.shape[0] // 5)))  # 根据样本量动态调整分箱数，至少 2 箱，避免样本过少时报错
    disc = KBinsDiscretizer(n_bins=actual_bins, encode="ordinal", strategy="quantile")  # 创建基于分位数的离散化器
    X_disc = pd.DataFrame(disc.fit_transform(X_imp), columns=feat_names, index=X.index).astype(int)  # 将特征离散化并转为整数

    relevance = {}  # 创建字典保存每个特征与标签之间的互信息
    for f in feat_names:  # 遍历每个特征
        relevance[f] = discrete_mi(X_disc[f].values, y_enc)  # 计算该特征与标签的互信息

    rel_df = pd.DataFrame({"feature": feat_names, "relevance_MI_to_y": [relevance[f] for f in feat_names]})  # 整理相关性结果
    rel_df = rel_df.sort_values("relevance_MI_to_y", ascending=False)  # 按互信息从高到低排序
    rel_df.to_csv(OUTPUT_DIR / "06_mRMR_relevance.csv", index=False, encoding=ENCODING)  # 保存特征与标签互信息表

    k = int(min(max(1, top_k), len(feat_names)))  # 限制最终选择特征数量不能小于 1 且不能超过总特征数
    selected = []  # 用于保存已选择特征
    remaining = set(feat_names)  # 用于保存尚未选择的候选特征
    selection_records = []  # 用于记录每一步选择过程

    def avg_redundancy(f, selected_list):  # 定义计算某个候选特征与已选特征集合平均冗余度的内部函数
        if not selected_list:  # 如果当前还没有已选特征
            return 0.0  # 冗余度设为 0
        f_vec = X_disc[f].values  # 取出候选特征离散化后的向量
        r_vals = []  # 创建列表记录与每个已选特征的互信息
        for s in selected_list:  # 遍历已选特征
            s_vec = X_disc[s].values  # 取出已选特征的离散向量
            r_vals.append(discrete_mi(f_vec, s_vec))  # 计算候选特征与该已选特征之间的互信息
        return float(np.mean(r_vals))  # 返回平均冗余度

    for step in range(k):  # 逐步贪婪选择 top-k 个特征
        best_f = None  # 初始化本轮最佳特征名
        best_score = -np.inf  # 初始化本轮最佳得分为负无穷
        best_D = None  # 初始化最佳相关度
        best_R = None  # 初始化最佳冗余度

        for f in remaining:  # 遍历所有未被选择的候选特征
            D = relevance[f]  # 获取该特征与标签的相关度
            R = avg_redundancy(f, selected)  # 获取该特征相对当前已选集合的平均冗余度

            if criterion.upper() == "MIQ":  # 如果选择 MIQ 准则
                score = D / (R + EPS)  # 按 MIQ = D / R 计算得分
            else:  # 默认使用 MID 准则
                score = D - R  # 按 MID = D - R 计算得分

            if score > best_score:  # 如果当前候选特征得分优于本轮最佳得分
                best_score = score  # 更新最佳得分
                best_f = f  # 更新最佳特征
                best_D = D  # 更新最佳相关度
                best_R = R  # 更新最佳冗余度

        if best_f is None:  # 如果本轮没有找到特征
            break  # 提前结束循环

        selected.append(best_f)  # 将最佳特征加入已选集合
        remaining.remove(best_f)  # 将该特征从候选集合中移除
        selection_records.append([step + 1, best_f, best_D, best_R, best_score])  # 记录本轮选择信息

    with open(OUTPUT_DIR / "07_mRMR_selected_features.txt", "w", encoding=ENCODING) as f:  # 以写入方式打开特征名输出文件
        for fea in selected:  # 遍历所有已选择特征
            f.write(f"{fea}\n")  # 每行写入一个特征名

    selection_df = pd.DataFrame(selection_records, columns=["rank", "feature", "relevance_D", "redundancy_R", "score"])  # 将选择过程整理成表格
    selection_df.to_csv(OUTPUT_DIR / "08_mRMR_selection_process.csv", index=False, encoding=ENCODING)  # 保存 mRMR 选择过程
    X_selected = X_imp[selected].copy()  # 取出最终 mRMR 选择后的特征矩阵
    X_selected.to_csv(OUTPUT_DIR / "09_mRMR_selected_feature_values.csv", index=False, encoding=ENCODING)  # 保存最终选中特征的特征值

    print(f"mRMR ({criterion.upper()}) 最终选择特征数: {len(selected)}")  # 打印最终选择特征数量

    return selected, rel_df, selection_df  # 返回最终特征列表、互信息表和选择过程表

def save_final_dataset(df: pd.DataFrame, y: pd.Series, X_final: pd.DataFrame):  # 定义保存最终数据集的函数
    id_part = df.iloc[:, :FEATURE_START_COL_INDEX].copy()  # 提取前两列，通常为 ID 列和标签列
    final_df = pd.concat([id_part, X_final], axis=1)  # 将 ID+标签 与最终特征拼接
    final_df.to_csv(OUTPUT_DIR / "10_Final_selected_dataset.csv", index=False, encoding=ENCODING)  # 保存最终建模数据集
    print(f"最终建模数据集已保存，维度: {final_df.shape}")  # 打印最终数据集维度

def main():  # 定义主函数
    df, X, y = load_and_prepare_data(INPUT_CSV)  # 读取并预处理数据

    X_mwu, p_series, sig_series = mann_whitney_filter(X, y)  # 第一步：进行 Mann-Whitney U 检验初筛
    if X_mwu.shape[1] == 0:  # 如果 MWU 后没有保留下任何特征
        sys.exit("[ERROR] Mann-Whitney U 检验后没有保留任何特征，请检查数据或放宽 p 值阈值。")  # 终止程序并提示

    X_spearman, drop_records, corr_matrix = spearman_reduction(X_mwu, p_series)  # 第二步：Spearman 去冗余
    if X_spearman.shape[1] == 0:  # 如果 Spearman 去冗余后没有特征
        sys.exit("[ERROR] Spearman 去冗余后没有保留任何特征，请检查阈值设置。")  # 终止程序并提示

    selected_features, rel_df, selection_df = mrmr_select(X_spearman, y, top_k=TOP_K, criterion=MRMR_CRITERION)  # 第三步：mRMR 终筛
    if len(selected_features) == 0:  # 如果 mRMR 没有选出特征
        sys.exit("[ERROR] mRMR 未能选出任何特征，请检查输入数据。")  # 终止程序并提示

    X_final = X_spearman[selected_features].copy()  # 按最终选择特征提取特征矩阵
    save_final_dataset(df, y, X_final)  # 保存最终建模数据集

    summary_df = pd.DataFrame({  # 创建流程总结表
        "step": ["Original", "After_MWU", "After_Spearman", "After_mRMR"],  # 各步骤名称
        "feature_count": [X.shape[1], X_mwu.shape[1], X_spearman.shape[1], X_final.shape[1]]  # 各步骤对应特征数量
    })  # 完成总结表构建
    summary_df.to_csv(OUTPUT_DIR / "11_feature_count_summary.csv", index=False, encoding=ENCODING)  # 保存各步骤特征数汇总

    print("\n[OK] 全流程完成。")  # 打印完成提示
    print("输出文件如下：")  # 打印输出标题
    print("01_MWU_p_values.csv                     -> 全部特征的 Mann-Whitney U 检验 p 值")  # 输出文件说明
    print("02_MWU_significant_features.csv         -> MWU 显著特征列表")  # 输出文件说明
    print("03_Spearman_correlation_matrix.csv      -> Spearman 相关矩阵")  # 输出文件说明
    print("04_Spearman_dropped_features.csv        -> Spearman 去冗余删除记录")  # 输出文件说明
    print("05_Features_after_Spearman.csv          -> Spearman 后特征值表")  # 输出文件说明
    print("06_mRMR_relevance.csv                   -> 各特征与标签的互信息")  # 输出文件说明
    print("07_mRMR_selected_features.txt           -> 最终选中特征名称")  # 输出文件说明
    print("08_mRMR_selection_process.csv           -> mRMR 每一步选择过程")  # 输出文件说明
    print("09_mRMR_selected_feature_values.csv     -> mRMR 最终特征值表")  # 输出文件说明
    print("10_Final_selected_dataset.csv           -> 最终建模数据集（前两列+最终特征）")  # 输出文件说明
    print("11_feature_count_summary.csv            -> 各步骤特征数量汇总")  # 输出文件说明

if __name__ == "__main__":  # 如果当前脚本作为主程序执行
    main()  # 调用主函数开始运行