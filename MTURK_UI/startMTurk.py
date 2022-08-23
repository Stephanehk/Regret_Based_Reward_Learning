import boto3
import xmltodict
import json
import pickle

create_hits_in_production = False #use Sandbox for testing and production for actually giving workers jobs
environments = {
  "production": {
    "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
    "preview": "https://www.mturk.com/mturk/preview"
  },
  "sandbox": {
    "endpoint":
          "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
    "preview": "https://workersandbox.mturk.com/mturk/preview"
  },
}
mturk_environment = environments["production"] if create_hits_in_production else environments["sandbox"]
aws_access_key_id = 'YOUR ACCESS KEY'
aws_secret_access_key = 'YOUR SECRET ACCESS KEY'

session = boto3.Session(profile_name='stephane.hk')
client = session.client(
    service_name='mturk',
    region_name='us-east-1',
    endpoint_url=mturk_environment['endpoint'],
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)


QUESTION_XML = open("interface.xml", "r")
question_xml = QUESTION_XML.read()

# question_xml = QUESTION_XML.format(html_layout)
#Regarding the requirements, we are using: location in US; num of HITs approved is greater than or equal to 50; total approval rate is greater than or equal to 99%.

TaskAttributes = {
    'MaxAssignments': 30,
    # How long the task will be available on MTurk (12 hours)
    'LifetimeInSeconds': 60*60*1,
    # How long Workers have to complete each item (60 minutes)
    'AssignmentDurationInSeconds': 60*60,
    # The reward you will offer Workers for each response
    'Reward': '5.00',
    'Title': 'Choose the image that shows better behavior',
    'Keywords': 'trajectory',
    'Description': 'Choose the image that shows better behavior',



}
#
# "QualificationRequirements":[{'QualificationTypeId':'00000000000000000040',
#                                'Comparator': 'GreaterThanOrEqualTo',
#                                'IntegerValues':[50]},
#                             {'QualificationTypeId':'000000000000000000L0',
#                                 'Comparator': 'GreaterThanOrEqualTo',
#                                 'IntegerValues':[99]},
#                               {"QualificationTypeId":"00000000000000000071",
#                                     "Comparator":"EqualTo",
#                                     "LocaleValues":[{
#                                         "Country":"US"
#                                 }]}]


results = []
hit_type_id = ''
response = client.create_hit(
    **TaskAttributes,
    Question=question_xml,
)
hit_type_id = response['HIT']['HITTypeId']
results.append({
    'hit_id': response['HIT']['HITId']
})

print("You can view the HITs here:")
print(mturk_environment['preview']+"?groupId={}".format(hit_type_id))
with open('results.pickle', 'wb') as temp:
    pickle.dump(results, temp)
print ("HIT ID: " + str(hit_type_id))
