"""
Module: store.views

Contains Django views for managing items, profiles, and deliveries in the store application.

Classes handle product listing, creation, updating, deletion, and delivery management.
The module integrates with Django's authentication and querying functionalities.
"""
import operator
from functools import reduce
from django.shortcuts import render
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import (
    DetailView,
    CreateView,
    UpdateView,
    DeleteView
)
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django_tables2 import SingleTableView
import django_tables2 as tables
from django_tables2.export.views import ExportMixin
from django_tables2.export.export import TableExport
from django.db.models import Q, Count, Sum, Avg
from django.views.generic.edit import FormMixin

from accounts.models import Profile, Vendor
from transactions.models import Sale
from .models import Category, Item, Delivery
from .forms import ProductForm
from .tables import ItemTable
# from keras.models import load_model
# import numpy as np
# import pandas as pd
# from datetime import datetime, timedelta,date
# import tensorflow as tf
# #import MinMaxScaler and create a new dataframe for LSTM model
# from sklearn.preprocessing import MinMaxScaler

@login_required
def dashboard(request):
    """
    View function to render the dashboard with item and profile data.

    Args:
    - request: HttpRequest object.

    Returns:
    - Rendered template with dashboard data.
    """
    
    # items=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/items.csv")
    # shops=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/shops.csv")
    # cats=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/item_categories.csv")
    # test=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/test - Copy.csv")
    # train=pd.read_csv("C:/Users/ankur/OneDrive/Desktop/dibba/drydata/sales_train.csv")

    # # Load the model
    # model = load_model('C:/Users/ankur/OneDrive/Desktop/dibba/saved_model.hdf5')
    
    profiles =  Profile.objects.all()
    Category.objects.annotate(nitem=Count('item'))
    items = Item.objects.all()
    total_items = Item.objects.all().aggregate(Sum('quantity')).get('quantity__sum', 0.00)
    items_count = items.count()
    profiles_count = profiles.count()
    categories = Category.objects.annotate(nitem=Count('item'))
    category_names = [category.name for category in categories]
    category_item_counts = [category.nitem for category in categories]

    #profile pagination
    page = request.GET.get('page', 1)
    paginator = Paginator(profiles, 3)
    try:
        profiles = paginator.page(page)
    except PageNotAnInteger:
        profiles = paginator.page(1)
    except EmptyPage:
        profiles = paginator.page(paginator.num_pages)

    #items pagination
    page = request.GET.get('page', 1)
    paginator = Paginator(items, 4)
    try:
        items = paginator.page(page)
    except PageNotAnInteger:
        items = paginator.page(1)
    except EmptyPage:
        items = paginator.page(paginator.num_pages)

    
    # train = train[(train.item_price < 100000 )& (train.item_cnt_day < 1000)]
    # train = train[(train.item_price > 0) & (train.item_cnt_day > 0)].reset_index(drop = True)


    # train['date'] = pd.to_datetime(train['date'], format="%d.%m.%Y")

    # #represent month in date field as its first day
    # train['date'] = train['date'].dt.year.astype('str') + '-' + train['date'].dt.month.astype('str') + '-01'
    # train['date'] = pd.to_datetime(train['date'])
    # #groupby date and sum the sales
    # train = train.groupby('date').item_cnt_day.sum().reset_index()


    # train.rename(columns = {'item_cnt_day':'item_cnt_month'}, inplace = True)

    # df_diff = train.copy()
    # #add previous sales to the next row
    # df_diff['prev_item_cnt_month'] = df_diff['item_cnt_month'].shift(1)
    # #drop the null values and calculate the difference
    # df_diff = df_diff.dropna()
    # df_diff['diff'] = (df_diff['item_cnt_month'] - df_diff['prev_item_cnt_month'])
    # df_diff.head(10)

    # #create dataframe for transformation from time series to supervised
    # df_supervised = df_diff.drop(['prev_item_cnt_month'],axis=1)
    # #adding lags
    # for inc in range(1,10):
    #     field_name = 'lag_' + str(inc)
    #     df_supervised[field_name] = df_supervised['diff'].shift(inc)
    # #drop null values
    # df_supervised = df_supervised.dropna().reset_index(drop=True)
    # df_supervised.head()

    # df_model = df_supervised.drop(['item_cnt_month','date'],axis=1)
    # #split train and test set
    # train_set, test_set = df_model[0:-6].values, df_model[-6:].values

    # #apply Min Max Scaler
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = scaler.fit(train_set)
    # # reshape test set
    # test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    # test_set_scaled = scaler.transform(test_set)


    # X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # y_pred = model.predict(X_test,batch_size=1)

    # #reshape y_pred
    # y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
    # #rebuild test set for inverse transform
    # pred_test_set = []
    # for index in range(0,len(y_pred)):
    #     print (np.concatenate([y_pred[index],X_test[index]],axis=1))
    #     pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
    # #reshape pred_test_set
    # pred_test_set = np.array(pred_test_set)
    # pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    # #inverse transform
    # pred_test_set_inverted = scaler.inverse_transform(pred_test_set)

    # #create dataframe that shows the predicted sales
    # result_list = []
    # sales_dates = list(train[-10:].date)
    # act_sales = list(train[-10:].item_cnt_month)
    # for index in range(0,len(pred_test_set_inverted)):
    #     result_dict = {}
    #     result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    #     result_dict['date'] = sales_dates[index+1]
    #     result_list.append(result_dict)
    # df_result = pd.DataFrame(result_list)
    # #for multistep prediction, replace act_sales with the predicted sales

    # df_result_copy= df_result.copy()
    # df_result_copy=df_result_copy.set_index('date')
    context = {
        
        'items': items,
        'profiles' : profiles,
        'profiles_count': profiles_count,
        'items_count': items_count,
        'total_items': total_items,
        'vendors' : Vendor.objects.all(),
        'delivery': Delivery.objects.all(),
        'sales': Sale.objects.all(),
        'category_names': category_names,
        'category_item_counts': category_item_counts,
        # 'result': df_result_copy
    }
    return render(request, 'store/dashboard.html', context)
class ProductListView(LoginRequiredMixin, ExportMixin, tables.SingleTableView):
    """
    View class to display a list of products.

    Attributes:
    - model: The model associated with the view.
    - table_class: The table class used for rendering.
    - template_name: The HTML template used for rendering the view.
    - context_object_name: The variable name for the context object.
    - paginate_by: Number of items per page for pagination.
    """
    model = Item
    table_class = ItemTable
    template_name = 'store/productslist.html'
    context_object_name = 'items'
    paginate_by = 10
    SingleTableView.table_pagination = False

class ItemSearchListView(ProductListView):
    """
    View class to search and display a filtered list of items.

    Attributes:
    - paginate_by: Number of items per page for pagination.
    """
    paginate_by = 10

    def get_queryset(self):
        result = super(ItemSearchListView, self).get_queryset()

        query = self.request.GET.get('q')
        if query:
            query_list = query.split()
            result = result.filter(
                reduce(operator.and_,
                       (Q(name__icontains=q) for q in query_list))
            )
        return result

class ProductDetailView(LoginRequiredMixin, FormMixin, DetailView):
    """
    View class to display detailed information about a product.

    Attributes:
    - model: The model associated with the view.
    - template_name: The HTML template used for rendering the view.
    """
    model = Item
    template_name = 'store/productdetail.html'

    def get_success_url(self):
        return reverse('product-detail', kwargs={'slug': self.object.slug})

class ProductCreateView(LoginRequiredMixin, CreateView):
    """
    View class to create a new product.

    Attributes:
    - model: The model associated with the view.
    - template_name: The HTML template used for rendering the view.
    - form_class: The form class used for data input.
    - success_url: The URL to redirect to upon successful form submission.
    """
    model = Item
    template_name = 'store/productcreate.html'
    form_class = ProductForm
    success_url = '/products'

    def test_func(self):
        #item = Item.objects.get(id=pk)
        if self.request.POST.get("quantity") < 1:
            return False
        else:
            return True

class ProductUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    """
    View class to update product information.

    Attributes:
    - model: The model associated with the view.
    - template_name: The HTML template used for rendering the view.
    - fields: The fields to be updated.
    - success_url: The URL to redirect to upon successful form submission.
    """
    model = Item
    template_name = 'store/productupdate.html'
    fields = ['name','category','quantity','selling_price', 'expiring_date', 'vendor']
    success_url = '/products'

    def test_func(self):
        if self.request.user.is_superuser:
            return True
        else:
            return False


class ProductDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    """
    View class to delete a product.

    Attributes:
    - model: The model associated with the view.
    - template_name: The HTML template used for rendering the view.
    - success_url: The URL to redirect to upon successful deletion.
    """
    model = Item
    template_name = 'store/productdelete.html'
    success_url = '/products'


    def test_func(self):
        if self.request.user.is_superuser:
            return True
        else:
            return False

class DeliveryListView(LoginRequiredMixin, ExportMixin, tables.SingleTableView):
    """
    View class to display a list of deliveries.

    Attributes:
    - model: The model associated with the view.
    - pagination: Number of items per page for pagination.
    - template_name: The HTML template used for rendering the view.
    - context_object_name: The variable name for the context object.
    """
    model = Delivery
    pagination = 10
    template_name = 'store/deliveries.html'
    context_object_name = 'deliveries'

class DeliverySearchListView(DeliveryListView):
    """
    View class to search and display a filtered list of deliveries.

    Attributes:
    - paginate_by: Number of items per page for pagination.
    """
    paginate_by = 10

    def get_queryset(self):
        result = super(DeliverySearchListView, self).get_queryset()

        query = self.request.GET.get('q')
        if query:
            query_list = query.split()
            result = result.filter(
                reduce(operator.and_,
                       (Q(customer_name__icontains=q) for q in query_list))
            )
        return result

class DeliveryDetailView(LoginRequiredMixin, DetailView):
    """
    View class to display detailed information about a delivery.

    Attributes:
    - model: The model associated with the view.
    - template_name: The HTML template used for rendering the view.
    """
    model = Delivery
    template_name = 'store/deliverydetail.html'
class DeliveryCreateView(LoginRequiredMixin, CreateView):
    """
    View class to create a new delivery.

    Attributes:
    - model: The model associated with the view.
    - fields: The fields to be included in the form.
    - template_name: The HTML template used for rendering the view.
    - success_url: The URL to redirect to upon successful form submission.
    """
    model = Delivery
    fields = ['item', 'customer_name', 'phone_number', 'location', 'date','is_delivered']
    template_name = 'store/deliveriescreate.html'
    success_url = '/deliveries'

class DeliveryUpdateView(LoginRequiredMixin, UpdateView):
    """
    View class to update delivery information.

    Attributes:
    - model: The model associated with the view.
    - fields: The fields to be updated.
    - template_name: The HTML template used for rendering the view.
    - success_url: The URL to redirect to upon successful form submission.
    """
    model = Delivery
    fields = ['item', 'customer_name', 'phone_number', 'location', 'date','is_delivered']
    template_name = 'store/deliveryupdate.html'
    success_url = '/deliveries'

class DeliveryDeleteView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    """
    View class to delete a delivery.

    Attributes:
    - model: The model associated with the view.
    - template_name: The HTML template used for rendering the view.
    - success_url: The URL to redirect to upon successful deletion.
    """
    model = Delivery
    template_name = 'store/productdelete.html'
    success_url = '/deliveries'

    def test_func(self):
        if self.request.user.is_superuser:
            return True
        else:
            return False
