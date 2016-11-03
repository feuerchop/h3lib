from utils.dataset_helper import *

event_json = '../../fixed2-mlTestCampaignCuris1.json'

print '#1: test get_data_batch... '
res = get_data_batch(event_json)
json.dump(res, open('temp.json', 'w'))
print 'templates used: ', res['meta']['templates_used']

print '#2: test get_data_set... '
d = get_data_set(event_json)
