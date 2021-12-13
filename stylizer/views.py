from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import SolvedSerializer, BlobSerializer
from .models import Solved, TestBlob

from .utils.stylizer import stylizer
from .utils.stylizerTEST import stylizerTEST


@api_view(['GET'])
def apiView(request):
    routes = [
        {'GET': '/api/stylizer/'},
        {'GET': '/api/stylizer/solved/id/'},
        {'POST': '/api/stylizer/upload/'},
    ]

    return Response(routes)

@api_view(['POST'])
def processView(request):
    json = request.data
    uid = json['uid']
    origin_url = json['origin']
    style_url = json['style']

    # TODO: process
    path_to_solved = stylizer(uid, origin_url, style_url)

    # TODO: save to solved
    solved = Solved.objects.create(
        uid=uid,
        stylized=path_to_solved,
    )

    serializer = SolvedSerializer(solved, many=False)
    return Response(serializer.data)

@api_view(['POST'])
def testView(request):
    json = request.data
    uid = json['uid']
    origin_url = json['origin']
    style_url = json['style']

    # TODO: process
    path_to_solved, load_elapse_t, render_elapse_t = stylizerTEST(uid, origin_url, style_url)

    # TODO: save to testblob
    blob = TestBlob.objects.create(
        uid = uid,
        stylized = path_to_solved,
        load_time = load_elapse_t,
        render_time = render_elapse_t,
    )

    serializer = BlobSerializer(blob, many=False)
    return Response(serializer.data)

# TODO: Retrive data

@api_view(['GET'])
def retrieveView(request, uid):
    solved = Solved.objects.get(uid=uid)
    serializer = SolvedSerializer(solved, many=False)

    return Response(serializer.data)
