from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import SolvedSerializer
from .models import Solved

from .utils.stylizer import stylizer


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

# TODO: Retrive data

@api_view(['GET'])
def retrieveView(request, uid):
    solved = Solved.objects.get(uid=uid)
    serializer = SolvedSerializer(solved, many=False)

    return Response(serializer.data)
