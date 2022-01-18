from django.shortcuts import render, redirect
from . import main

# Create your views here.
def sentiment(request):
    res = ""
    if request.method == 'POST':
        if request.POST.get('pred_button'):
            name = request.POST['text data']
            res = main.pre(str(name))
            # print(res)
        else:
            redirect('homepage')
            res = ""
    else:
        print("Error Occurred")

    return render(request, "index.html", {'result': res})