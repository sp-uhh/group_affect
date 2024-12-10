from readers.memo_ga import MEMOGroupAff
from readers.utils.utils import dict_from_json 
from readers.utils.video_utils import write_participants_cropped_videos


groupId = "4"
sessId = "1"

video_name = "group"+groupId+"_session"+sessId
memodb = MEMOGroupAff()
memodb.get_interactions_df()


feature_type = "vggish"
memodb.load_featureset(feature_type)


for interaction in memodb.interactions_df:
    for particip in interaction.participants:
        feat = particip.get_feature(feature_type)
        print(interaction.id, ": Participant ", particip.id, " -- ", feat.shape)



























# sample_video = memodb.videos_folder+video_name+".mp4"
# config = dict_from_json(memodb.config_folder+video_name+".json")



# write_participants_cropped_videos(config, sample_video, memodb.videos_folder, groupId, sessId)