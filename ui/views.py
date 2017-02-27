from django.shortcuts import render
from django.http import HttpResponse
from .models import Training_Input
from .sample import sample

context = {}

def index(request):
    save_dir_list = Training_Input.objects.order_by('training_input_name')
    global context
    if len(context) == 0:
        context = {'save_dir_list' : save_dir_list}
    args = {
    'save_dir':'ui/save/jurassicPark',
    'n':200,
    'prime':' ',
    'pick':1,
    'sample':1}
    if request.method == 'POST':
        if request.POST.get('submit'): #submit button
            if request.POST.get('select training input') != 'Custom':
                args['save_dir'] = request.POST.get('select training input')
            if request.POST.get('WordCount') and int(request.POST.get('WordCount')) > 0:
                args['n'] = int(request.POST.get('WordCount')) #set number of words to generate
            if 'result' in context.keys() and len(context['result']) > 0:
                args['prime'] = context['result'] #if there is already generated text, use it to prime the new output
            context['result'] = sample(args) #generate new output
        if request.POST.get('reset'): #remove output if reset button is pressed
            context['result'] = ''
    return render(request,'ui/index.html',context)
