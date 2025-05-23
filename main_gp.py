"""
以下のすべてのコードは、コードの簡易化と実行時間削減のため10時間*10回測定したデータのみを用いている。
その他、すべてのデータを用いる場合は、各関数を変更する必要がある。
"""

"""
2: gpflow_t_distribution.py t分布を仮定したGaussian Process Regression
"""
from gp.gpflow_t_distribution import main as gpflow_t_main
gpflow_t_main()

"""
3:linear.py 線形回帰を用いたもの(ガウス過程回帰ではない)
"""
from gp.linear import main as linear_main
linear_main()

"""
5:ITGP_robustgp.py ITGPを用いたもの
"""
from gp.ITGP_robustgp import main as ITGP_robust_main
ITGP_robust_main()