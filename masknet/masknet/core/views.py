from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from masknet.core import serializers
from rest_framework.parsers import FormParser, MultiPartParser
from django.core.files.storage import FileSystemStorage
# from masknet.core.mask_net import get_pred_mask
# from masknet.core.person_detect import get_person
from masknet.core.person_detect_yolo import get_persons
from datetime import datetime

# Create your views here.



class PredictAPI(GenericAPIView):
    serializer_class = serializers.PredictSerializer
    parser_classes = ((FormParser, MultiPartParser))

    def get(self, request, *args, **kwargs):
        result = dict()
        result['status'] = True
        result['message'] = "Please use post method to get Prediction"
        return Response(result)

    def post(self, request, *args , **kwargs):
        result = dict()
        s = self.get_serializer(data = request.data)
        if s.is_valid():
            result['status'] = True
            fs = FileSystemStorage()
            start = datetime.now()
            myfile = s.validated_data['input_img']
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.path(filename)
            resp = get_persons(uploaded_file_url)
            end = datetime.now()
            result['throughput'] = "{0} seconds".format((end - start).seconds)
            if len(resp) ==0:
                result['status'] = False
                result['message'] = "No Persons found!"
            else:
                result['result'] = resp
                result['message'] = "At least {0} persons found.".format(len(resp))
        else:
            result['status'] = False
            result['message'] = "Invalid File"
            result['errors'] = s.errors
            
        return Response(result)