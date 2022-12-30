from pettingzoo.test import api_test, parallel_api_test
from env.santorini.env.santorini import env

santorini_env = env()
api_test(santorini_env, num_cycles=1000, verbose_progress=True)
