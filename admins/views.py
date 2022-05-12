from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistrationModel
# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})




def AdminHome(request):
    return render(request, 'admins/AdminHome.html')

def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})

def AdminViewRNN(request):
    from users.utility import deep_learnig_model
    train_score, train_rmse, test_score, test_rmse = deep_learnig_model.calculate_rnn_results()
    return render(request, "admins/admin_rnn_results.html", {
        "train_score": train_score, "train_rmse": train_rmse, "test_score": test_score, "test_rmse": test_rmse
    })