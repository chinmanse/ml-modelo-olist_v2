from pydantic import BaseModel

class DelayRequest(BaseModel):
    customer_geolocation_lat: float
    order_purchase_timestamp_month: int
    freight_value: float
    customer_geolocation_lng: float
    customer_geolocation_zip_code_prefix: int
    customer_zip_code_prefix: int
    review_score: float
    seller_geolocation_lng: float
    seller_zip_code_prefix: int
    seller_geolocation_zip_code_prefix: int
    seller_geolocation_lat: float
    payment_value: float
    order_purchase_timestamp_year: int
    price: float
    product_weight_g: float
    order_purchase_timestamp_day: int
    product_height_cm: float
    product_width_cm: float
    product_length_cm: float
    order_purchase_timestamp_dayofweek: int