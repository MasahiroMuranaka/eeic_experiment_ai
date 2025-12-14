# プログラミングの原則まとめ

## DRY原則 (Don't Repeat Yourself)
**内容**:
コードやロジックの重複を避ける。同じ機能や情報を複数の場所で繰り返し記述するのではなく、一箇所にまとめて管理する。

**コード例**:
```python
# DRYに違反する例
def print_user1():
    print("名前: 山田太郎, 年齢: 30")

def print_user2():
    print("名前: 佐藤花子, 年齢: 25")

# DRYを守る例
def print_user(name, age):
    print(f"名前: {name}, 年齢: {age}")

print_user("山田太郎", 30)
print_user("佐藤花子", 25)
```

---

## S - 単一責任の原則 (Single Responsibility Principle)
**内容**:
1つのクラスやモジュールは、1つの責任だけを持つべきである。クラスが複数の役割を持つと、変更が必要になったときに影響範囲が広がりやすくなるのを防ぐ。

**コード例**:
```python
# 単一責任に違反する例
class User:
    def save_to_database(self):
        print("データベースに保存")

    def send_email(self):
        print("メール送信")

# 改善例
class User:
    def save(self):
        UserRepository().save(self)

class UserRepository:
    def save(self, user):
        print("データベースに保存")

class EmailService:
    def send_email(self, user):
        print("メール送信")
```

---

## O - オープン・クローズドの原則 (Open/Closed Principle)
**内容**:
クラスは拡張に対して開いており、修正に対して閉じているべき。既存のコードを変更せずに新しい機能を追加できるようにする。

**コード例**:
```python
# オープン・クローズドに違反する例
class Payment:
    def process(self, payment_type):
        if payment_type == "credit":
            print("クレジットカード処理")
        elif payment_type == "paypal":
            print("PayPal処理")

# 改善例（ポリモーフィズムを使用）
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self):
        pass

class CreditCard(PaymentMethod):
    def process(self):
        print("クレジットカード処理")

class PayPal(PaymentMethod):
    def process(self):
        print("PayPal処理")

class Payment:
    def process(self, method: PaymentMethod):
        method.process()
```

---

## L - リスコフの置換原則 (Liskov Substitution Principle)
**内容**:
派生クラスは基底クラスと置換可能でなければならない。継承を使用する際に、基底クラスの機能を壊さないようにする。

**コード例**:
```python
# リスコフに違反する例
class Bird:
    def fly(self):
        print("飛ぶ")

class Penguin(Bird):
    def fly(self):
        raise Exception("ペンギンは飛べません") # 基底クラスの期待を裏切る

# 改善例
class Bird:
    pass

class FlyingBird(Bird):
    def fly(self):
        print("飛ぶ")

class Penguin(Bird):
    pass
```

---

## I - インターフェース分離の原則 (Interface Segregation Principle)
**内容**:
クライアントが必要としないインターフェースを強制すべきではない。巨大なインターフェースを避け、必要な機能だけを提供する。

**コード例**:
```python
# インターフェース分離に違反する例
class Worker:
    def work(self):
        pass
    def eat(self):
        pass

class Robot(Worker):
    def work(self):
        print("働く")
    def eat(self):
        # ロボットは食べない
        pass

# 改善例
class Workable:
    def work(self):
        pass

class Eatable:
    def eat(self):
        pass

class Human(Workable, Eatable):
    def work(self):
        print("働く")
    def eat(self):
        print("食べる")

class Robot(Workable):
    def work(self):
        print("働く")
```

---

## D - 依存性逆転の原則 (Dependency Inversion Principle)
**内容**:
高レベルのモジュールは低レベルのモジュールに依存せず、両者は抽象に依存すべき。依存関係を逆転させ、柔軟性とテスト容易性を高める。

**コード例**:
```python
# 依存性逆転に違反する例
class Database:
    def save(self):
        print("データベースに保存")

class UserService:
    def __init__(self):
        self.db = Database() # 具体クラスに依存

# 改善例
from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def save(self):
        pass

class Database(Storage):
    def save(self):
        print("データベースに保存")

class UserService:
    def __init__(self, storage: Storage):
        self.storage = storage # 抽象に依存
```

---

## KISS原則 (Keep It Simple, Stupid)
**内容**:
「シンプルに保て、馬鹿者」という意味で、設計や実装を必要以上に複雑にしないことを推奨。シンプルなコードは理解しやすく、バグが少なく、保守が容易。

**コード例**:
```python
# 複雑すぎる例
def calculate_price(item, discount, tax, is_premium):
    return item * (1 - discount) * (1 + tax) if is_premium else item * (1 + tax)

# KISSを適用
def calculate_price(item, discount, tax):
    price = item * (1 - discount)
    return price * (1 + tax)
```

---

## YAGNI原則 (You Aren't Gonna Need It)
**内容**:
「それは必要ないよ」という意味で、現在の要件に必要ない機能やコードを追加しない。無駄な開発時間を減らし、コードベースを小さく保つ。

**コード例**:
```python
# YAGNIに違反
class UserManager:
    def save_to_db(self):
        pass
    def save_to_file(self):
        pass # 今は使わない
    def save_to_cloud(self):
        pass # 今は使わない

# YAGNIを適用
class UserManager:
    def save_to_db(self):
        pass # 今必要な機能だけ
```

---

## Law of Demeter (デメテルの法則)
**内容**:
「知り合い以外とは話さない」という比喩で、オブジェクトが他のオブジェクトの内部構造に深く依存しないようにする。メソッドは自分自身のフィールドや引数として渡されたオブジェクトのメソッドのみを呼び出す。

**コード例**:
```python
# デメテルの法則に違反
class Order:
    def get_customer(self):
        return self.customer

class Customer:
    def get_address(self):
        return self.address

order.get_customer().get_address() # 深い依存

# 改善例
class Order:
    def get_customer_address(self):
        return self.customer.get_address()
```

---

## PoLA (Principle of Least Astonishment)
**内容**:
「驚き最小の原則」。コードやインターフェースが直感的で、ユーザーの期待を裏切らないようにする。使いやすさと予測可能性を高める。

**コード例**:
```python
# 驚く例
def add(a, b):
    return str(a) + str(b) # "23"

# 期待通り
def add(a, b):
    return a + b # 5
```

---

## TDAE (Tell, Don't Ask)
**内容**:
「聞かずに命令しろ」。オブジェクトの状態を問い合わせるのではなく、直接命令を送る。カプセル化を強化し、制御の逆転を防ぐ。

**コード例**:
```python
# Ask（聞く）
class Light:
    def is_on(self):
        return self.state
    def turn_on(self):
        self.state = True

if light.is_on():
    pass
else:
    light.turn_on()

# Tell（命令）
class Light:
    def ensure_on(self):
        self.state = True

light.ensure_on()
```