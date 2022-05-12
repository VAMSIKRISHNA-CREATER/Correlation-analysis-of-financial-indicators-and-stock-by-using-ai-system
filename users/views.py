from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['logged user'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def user_view_data(request):
    import pandas as pd
    from django.conf import settings
    import os
    path =os.path.join(settings.MEDIA_ROOT, 'ada.csv')
    df = pd.read_csv(path)
    df = df[['Date','Open', 'High', 'Low', 'Close']]
    # df = df.head(10)
    # df = df.head(100).to_html
    df = df.to_html

    return render(request, 'users/view_data.html',{'data': df})


def user_ml(request):
    from .utility import processml
    mae, mse, r2 = processml.calc_linear_regression()
    rf_mae, rf_mse, rf_r2 = processml.calc_random_forest_regressor()
    svm_mae, svm_mse, svm_r2 = processml.calc_svm()
    lr = {'mae': mae,'mse': mse,"r2": r2}
    rf = {'rf_mae': rf_mae, 'rf_mse': rf_mse, "rf_r2": rf_r2}
    svm = {'svm_mae': svm_mae, 'svm_mse': svm_mse, "svm_r2": svm_r2}
    return render(request,'users/ml_result.html',{"lr": lr, "rf": rf, "svm":svm} )

def user_deep_learning(request):
    from .utility import deep_learnig_model
    train_score,train_rmse,test_score,test_rmse = deep_learnig_model.calculate_rnn_results()
    return render(request, "users/rnn_results.html", {
        "train_score": train_score, "train_rmse": train_rmse, "test_score": test_score, "test_rmse": test_rmse
    })

def user_forecast(request):
    from .utility.ForeCastModel import StockPriceForeCast
    obj = StockPriceForeCast()
    rslt = obj.start_future_prediction()
    rslt = rslt.to_html
    return render(request,"users/forecast_result.html", {"rslt": rslt})