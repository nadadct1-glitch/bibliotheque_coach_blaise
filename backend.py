"""
API FastAPI pour la gestion de commerce de céréales
Fichier unique avec PostgreSQL direct
Version: 1.0.0

Installation:
pip install fastapi uvicorn psycopg2-binary python-multipart

Exécution:
python main.py

Configuration:
Modifier les variables DATABASE_CONFIG ci-dessous
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import uvicorn


# ============================================================================
# CONFIGURATION DE LA BASE DE DONNÉES
# ============================================================================
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "cereales_db",
    "user": "postgres",
    "password": "beauty"
}

# Pool de connexions
db_pool = None

def init_db_pool():
    """Initialiser le pool de connexions"""
    global db_pool
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=20,
        **DATABASE_CONFIG
    )

@contextmanager
def get_db_connection():
    """Gestionnaire de contexte pour les connexions DB"""
    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        db_pool.putconn(conn)


from decimal import Decimal

def convert_decimal_to_float(data):
    """Convertit tous les Decimal en float dans un dict ou une liste"""
    if isinstance(data, dict):
        return {k: convert_decimal_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_decimal_to_float(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data
    
# ============================================================================
# CRÉATION DES TABLES
# ============================================================================
def create_tables():
    """Créer toutes les tables nécessaires"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Table Ventes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ventes (
                id SERIAL PRIMARY KEY,
                receipt_number VARCHAR(50) UNIQUE NOT NULL,
                
                -- Client info
                customer_name VARCHAR(255),
                customer_phone VARCHAR(20),
                
                -- Product details
                product_type VARCHAR(100) NOT NULL,
                unit_type VARCHAR(10) NOT NULL CHECK (unit_type IN ('bol', 'sac')),
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                sac_size INTEGER CHECK (sac_size IN (20, 40, 60, 80)),
                
                -- Pricing
                unit_price NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
                discount NUMERIC(5, 2) DEFAULT 0 CHECK (discount >= 0 AND discount <= 100),
                subtotal NUMERIC(10, 2) NOT NULL,
                discount_amount NUMERIC(10, 2) DEFAULT 0,
                total_amount NUMERIC(10, 2) NOT NULL CHECK (total_amount >= 0),
                
                -- Payment
                payment_method VARCHAR(20) NOT NULL CHECK (payment_method IN ('cash', 'mobile', 'bank', 'credit', 'advance')),
                advance_amount NUMERIC(10, 2),
                
                -- Date and time
                sale_date TIMESTAMP WITH TIME ZONE NOT NULL,
                sale_time VARCHAR(10) NOT NULL,
                
                -- Notes
                notes TEXT,
                
                -- Metadata
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE,
                
                CONSTRAINT check_advance_amount CHECK (
                    (payment_method != 'advance') OR (advance_amount IS NOT NULL AND advance_amount > 0)
                )
            );
            
            CREATE INDEX IF NOT EXISTS idx_ventes_receipt ON ventes(receipt_number);
            CREATE INDEX IF NOT EXISTS idx_ventes_date ON ventes(sale_date);
            CREATE INDEX IF NOT EXISTS idx_ventes_product ON ventes(product_type);
        """)
        
        # Table Dépenses
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS depenses (
                id SERIAL PRIMARY KEY,
                expense_number VARCHAR(50) UNIQUE NOT NULL,
                
                -- Expense details
                expense_amount NUMERIC(10, 2) NOT NULL CHECK (expense_amount > 0),
                expense_category VARCHAR(50) NOT NULL CHECK (expense_category IN (
                    'salaires', 'loyer', 'electricite', 'eau', 'telephone',
                    'transport', 'maintenance', 'fournitures', 'impots',
                    'assurance', 'publicite', 'autre'
                )),
                expense_reason TEXT NOT NULL,
                
                -- Beneficiary & Authorization
                beneficiary VARCHAR(255) NOT NULL,
                cashier VARCHAR(255) NOT NULL,
                authorized_by VARCHAR(255) NOT NULL,
                
                -- Cash status
                cash_before NUMERIC(10, 2) NOT NULL CHECK (cash_before >= 0),
                cash_after NUMERIC(10, 2) NOT NULL CHECK (cash_after >= 0),
                
                -- Date and time
                expense_date TIMESTAMP WITH TIME ZONE NOT NULL,
                expense_time VARCHAR(10) NOT NULL,
                
                -- Notes
                notes TEXT,
                
                -- Metadata
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE,
                
                CONSTRAINT check_cash_calculation CHECK (cash_after = cash_before - expense_amount)
            );
            
            CREATE INDEX IF NOT EXISTS idx_depenses_number ON depenses(expense_number);
            CREATE INDEX IF NOT EXISTS idx_depenses_date ON depenses(expense_date);
            CREATE INDEX IF NOT EXISTS idx_depenses_category ON depenses(expense_category);
        """)
        
        # Table Inventaire
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventaire (
                id SERIAL PRIMARY KEY,
                
                -- Cereal type
                cereal_type VARCHAR(100) NOT NULL,
                variety VARCHAR(255),
                
                -- Quantity
                bowl_quantity INTEGER NOT NULL CHECK (bowl_quantity >= 0),
                sac_quantity INTEGER NOT NULL CHECK (sac_quantity >= 0),
                sac_type INTEGER NOT NULL CHECK (sac_type IN (20, 40, 60, 80)),
                total_bowls INTEGER NOT NULL,
                
                -- Pricing
                price_per_bowl NUMERIC(10, 2) NOT NULL CHECK (price_per_bowl > 0),
                price_per_sac NUMERIC(10, 2) NOT NULL,
                total_purchase_cost NUMERIC(10, 2) NOT NULL,
                
                -- Logistics
                arrival_date TIMESTAMP WITH TIME ZONE NOT NULL,
                origin VARCHAR(255) NOT NULL,
                transport_cost NUMERIC(10, 2) NOT NULL CHECK (transport_cost >= 0),
                grand_total NUMERIC(10, 2) NOT NULL,
                
                -- Notes
                notes TEXT,
                
                -- Stock status
                current_stock INTEGER NOT NULL CHECK (current_stock >= 0),
                is_depleted INTEGER DEFAULT 0 CHECK (is_depleted IN (0, 1)),
                
                -- Metadata
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE,
                
                CONSTRAINT check_total_bowls CHECK (total_bowls = (sac_quantity * sac_type) + bowl_quantity),
                CONSTRAINT check_grand_total CHECK (grand_total = total_purchase_cost + transport_cost)
            );
            
            CREATE INDEX IF NOT EXISTS idx_inventaire_cereal ON inventaire(cereal_type);
            CREATE INDEX IF NOT EXISTS idx_inventaire_date ON inventaire(arrival_date);
        """)
        
        cursor.close()

# ============================================================================
# ENUMS
# ============================================================================
class PaymentMethodEnum(str, Enum):
    CASH = "cash"
    MOBILE = "mobile"
    BANK = "bank"
    CREDIT = "credit"
    ADVANCE = "advance"

class UnitTypeEnum(str, Enum):
    BOL = "bol"
    SAC = "sac"

class ExpenseCategoryEnum(str, Enum):
    SALAIRES = "salaires"
    LOYER = "loyer"
    ELECTRICITE = "electricite"
    EAU = "eau"
    TELEPHONE = "telephone"
    TRANSPORT = "transport"
    MAINTENANCE = "maintenance"
    FOURNITURES = "fournitures"
    IMPOTS = "impots"
    ASSURANCE = "assurance"
    PUBLICITE = "publicite"
    AUTRE = "autre"

# ============================================================================
# SCHEMAS PYDANTIC - VENTES
# ============================================================================
class VenteCreate(BaseModel):
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    product_type: str = Field(..., min_length=1, max_length=100)
    unit_type: UnitTypeEnum
    quantity: int = Field(..., gt=0)
    sac_size: Optional[int] = Field(None, ge=20, le=80)
    unit_price: float = Field(..., ge=0)
    discount: float = Field(0.0, ge=0, le=100)
    payment_method: PaymentMethodEnum
    advance_amount: Optional[float] = Field(None, ge=0)
    sale_date: datetime
    sale_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    notes: Optional[str] = None

    @field_validator('advance_amount')
    @classmethod
    def validate_advance_amount(cls, v, info):
        payment_method = info.data.get('payment_method')
        if payment_method == PaymentMethodEnum.ADVANCE:
            if v is None or v <= 0:
                raise ValueError('advance_amount requis et positif pour paiement "advance"')
        return v

    @field_validator('sac_size')
    @classmethod
    def validate_sac_size(cls, v, info):
        unit_type = info.data.get('unit_type')
        if unit_type == UnitTypeEnum.SAC and v is None:
            raise ValueError('sac_size requis quand unit_type est "sac"')
        return v

class VenteResponse(BaseModel):
    id: int
    receipt_number: str
    customer_name: Optional[str]
    customer_phone: Optional[str]
    product_type: str
    unit_type: str
    quantity: int
    sac_size: Optional[int]
    unit_price: float
    discount: float
    subtotal: float
    discount_amount: float
    total_amount: float
    payment_method: str
    advance_amount: Optional[float]
    sale_date: datetime
    sale_time: str
    notes: Optional[str]
    created_at: datetime

# ============================================================================
# SCHEMAS PYDANTIC - DÉPENSES
# ============================================================================
class DepenseCreate(BaseModel):
    expense_amount: float = Field(..., gt=0)
    expense_category: ExpenseCategoryEnum
    expense_reason: str = Field(..., min_length=10)
    beneficiary: str = Field(..., min_length=1, max_length=255)
    cashier: str = Field(..., min_length=1, max_length=255)
    authorized_by: str = Field(..., min_length=1, max_length=255)
    cash_before: float = Field(..., ge=0)
    expense_date: datetime
    expense_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    notes: Optional[str] = None

    @field_validator('cash_before')
    @classmethod
    def validate_cash_before(cls, v, info):
        expense_amount = info.data.get('expense_amount')
        if expense_amount and v < expense_amount:
            raise ValueError('Caisse insuffisante: cash_before doit être >= expense_amount')
        return v

class DepenseResponse(BaseModel):
    id: int
    expense_number: str
    expense_amount: float
    expense_category: str
    expense_reason: str
    beneficiary: str
    cashier: str
    authorized_by: str
    cash_before: float
    cash_after: float
    expense_date: datetime
    expense_time: str
    notes: Optional[str]
    created_at: datetime

# ============================================================================
# SCHEMAS PYDANTIC - INVENTAIRE
# ============================================================================
class InventaireCreate(BaseModel):
    cereal_type: str = Field(..., min_length=1, max_length=100)
    variety: Optional[str] = None
    bowl_quantity: int = Field(..., ge=0)
    sac_quantity: int = Field(..., ge=0)
    sac_type: int = Field(..., ge=20, le=80)
    price_per_bowl: float = Field(..., gt=0)
    arrival_date: datetime
    origin: str = Field(..., min_length=1, max_length=255)
    transport_cost: float = Field(..., ge=0)
    notes: Optional[str] = None

    @field_validator('sac_type')
    @classmethod
    def validate_sac_type(cls, v):
        if v not in [20, 40, 60, 80]:
            raise ValueError('sac_type doit être: 20, 40, 60 ou 80')
        return v

class InventaireResponse(BaseModel):
    id: int
    cereal_type: str
    variety: Optional[str]
    bowl_quantity: int
    sac_quantity: int
    sac_type: int
    total_bowls: int
    price_per_bowl: float
    price_per_sac: float
    total_purchase_cost: float
    arrival_date: datetime
    origin: str
    transport_cost: float
    grand_total: float
    notes: Optional[str]
    current_stock: int
    is_depleted: int
    created_at: datetime

# ============================================================================
# SERVICES - VENTES
# ============================================================================
def generate_receipt_number() -> str:
    """Générer un numéro de reçu unique"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ventes")
        count = cursor.fetchone()[0]
        cursor.close()
        return f"REÇU-{str(count + 1).zfill(4)}"

def calculate_vente_amounts(quantity: int, unit_price: float, discount: float) -> dict:
    """Calculer les montants de la vente"""
    subtotal = quantity * unit_price
    discount_amount = (subtotal * discount) / 100
    total_amount = subtotal - discount_amount
    
    return {
        "subtotal": round(subtotal, 2),
        "discount_amount": round(discount_amount, 2),
        "total_amount": round(total_amount, 2)
    }

def create_vente(data: VenteCreate) -> dict:
    """Créer une nouvelle vente"""
    receipt_number = generate_receipt_number()
    amounts = calculate_vente_amounts(data.quantity, data.unit_price, data.discount)
    
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            INSERT INTO ventes (
                receipt_number, customer_name, customer_phone, product_type,
                unit_type, quantity, sac_size, unit_price, discount,
                subtotal, discount_amount, total_amount, payment_method,
                advance_amount, sale_date, sale_time, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING *
        """, (
            receipt_number, data.customer_name, data.customer_phone, data.product_type,
            data.unit_type.value, data.quantity, data.sac_size, data.unit_price,
            data.discount, amounts['subtotal'], amounts['discount_amount'],
            amounts['total_amount'], data.payment_method.value, data.advance_amount,
            data.sale_date, data.sale_time, data.notes
        ))
        
        result = cursor.fetchone()
        cursor.close()
        return dict(result)

def get_ventes(skip: int = 0, limit: int = 100, start_date: Optional[datetime] = None,
               end_date: Optional[datetime] = None, product_type: Optional[str] = None) -> List[dict]:
    """Obtenir la liste des ventes avec filtres"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM ventes WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND sale_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND sale_date <= %s"
            params.append(end_date)
        if product_type:
            query += " AND product_type = %s"
            params.append(product_type)
        
        query += " ORDER BY created_at DESC OFFSET %s LIMIT %s"
        params.extend([skip, limit])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in results]

def get_vente_by_id(vente_id: int) -> Optional[dict]:
    """Obtenir une vente par ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM ventes WHERE id = %s", (vente_id,))
        result = cursor.fetchone()
        cursor.close()
        return dict(result) if result else None


def get_daily_stats(date: datetime) -> dict:
    """Obtenir les statistiques du jour"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as sales_count,
                COALESCE(SUM(total_amount), 0) as total_revenue
            FROM ventes
            WHERE DATE(sale_date) = DATE(%s)
        """, (date,))
        
        result = cursor.fetchone()
        cursor.close()
        
        # CORRECTION: Convertir les Decimal en float avant calcul
        sales_count = int(result[0]) if result[0] else 0
        total_revenue = float(result[1]) if result[1] else 0.0
        
        return {
            "sales_count": sales_count,
            "total_revenue": total_revenue,
            "estimated_profit": total_revenue * 0.2  # Maintenant les deux sont float
        }
# ============================================================================
# SERVICES - DÉPENSES
# ============================================================================
def generate_expense_number() -> str:
    """Générer un numéro de dépense unique"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM depenses")
        count = cursor.fetchone()[0]
        cursor.close()
        return f"DEP-{str(count + 1).zfill(4)}"

def create_depense(data: DepenseCreate) -> dict:
    """Créer une nouvelle dépense"""
    expense_number = generate_expense_number()
    cash_after = round(data.cash_before - data.expense_amount, 2)
    
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            INSERT INTO depenses (
                expense_number, expense_amount, expense_category, expense_reason,
                beneficiary, cashier, authorized_by, cash_before, cash_after,
                expense_date, expense_time, notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING *
        """, (
            expense_number, data.expense_amount, data.expense_category.value,
            data.expense_reason, data.beneficiary, data.cashier, data.authorized_by,
            data.cash_before, cash_after, data.expense_date, data.expense_time, data.notes
        ))
        
        result = cursor.fetchone()
        cursor.close()
        return dict(result)

def get_depenses(skip: int = 0, limit: int = 100, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None, category: Optional[str] = None) -> List[dict]:
    """Obtenir la liste des dépenses avec filtres"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM depenses WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND expense_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND expense_date <= %s"
            params.append(end_date)
        if category:
            query += " AND expense_category = %s"
            params.append(category)
        
        query += " ORDER BY created_at DESC OFFSET %s LIMIT %s"
        params.extend([skip, limit])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in results]

def get_period_expenses(days: int = 1) -> dict:
    """Obtenir les statistiques des dépenses sur une période"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COALESCE(SUM(expense_amount), 0) as total
            FROM depenses
            WHERE expense_date >= CURRENT_TIMESTAMP - INTERVAL '%s days'
        """, (days,))
        
        total = cursor.fetchone()[0]
        cursor.close()
        return {"total_expenses": float(total)}

# ============================================================================
# SERVICES - INVENTAIRE
# ============================================================================
def calculate_inventaire_totals(data: InventaireCreate) -> dict:
    """Calculer les totaux de l'inventaire"""
    total_bowls = (data.sac_quantity * data.sac_type) + data.bowl_quantity
    price_per_sac = data.price_per_bowl * data.sac_type
    total_purchase_cost = (data.sac_quantity * price_per_sac) + (data.bowl_quantity * data.price_per_bowl)
    grand_total = total_purchase_cost + data.transport_cost
    
    return {
        "total_bowls": total_bowls,
        "price_per_sac": round(price_per_sac, 2),
        "total_purchase_cost": round(total_purchase_cost, 2),
        "grand_total": round(grand_total, 2),
        "current_stock": total_bowls
    }

def create_inventaire(data: InventaireCreate) -> dict:
    """Créer une nouvelle entrée d'inventaire"""
    totals = calculate_inventaire_totals(data)
    
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            INSERT INTO inventaire (
                cereal_type, variety, bowl_quantity, sac_quantity, sac_type,
                total_bowls, price_per_bowl, price_per_sac, total_purchase_cost,
                arrival_date, origin, transport_cost, grand_total, notes,
                current_stock, is_depleted
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING *
        """, (
            data.cereal_type, data.variety, data.bowl_quantity, data.sac_quantity,
            data.sac_type, totals['total_bowls'], data.price_per_bowl,
            totals['price_per_sac'], totals['total_purchase_cost'], data.arrival_date,
            data.origin, data.transport_cost, totals['grand_total'], data.notes,
            totals['current_stock'], 0
        ))
        
        result = cursor.fetchone()
        cursor.close()
        return dict(result)

def get_inventaires(skip: int = 0, limit: int = 100, cereal_type: Optional[str] = None,
                    in_stock_only: bool = False) -> List[dict]:
    """Obtenir la liste des inventaires avec filtres"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM inventaire WHERE 1=1"
        params = []
        
        if cereal_type:
            query += " AND cereal_type = %s"
            params.append(cereal_type)
        if in_stock_only:
            query += " AND is_depleted = 0 AND current_stock > 0"
        
        query += " ORDER BY created_at DESC OFFSET %s LIMIT %s"
        params.extend([skip, limit])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in results]

def update_stock(inventory_id: int, quantity_sold: int) -> dict:
    """Mettre à jour le stock après une vente"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Vérifier le stock actuel
        cursor.execute("SELECT current_stock FROM inventaire WHERE id = %s", (inventory_id,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError("Inventaire non trouvé")
        
        current_stock = result['current_stock']
        if current_stock < quantity_sold:
            raise ValueError("Stock insuffisant")
        
        new_stock = current_stock - quantity_sold
        is_depleted = 1 if new_stock == 0 else 0
        
        cursor.execute("""
            UPDATE inventaire
            SET current_stock = %s, is_depleted = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            RETURNING *
        """, (new_stock, is_depleted, inventory_id))
        
        result = cursor.fetchone()
        cursor.close()
        return dict(result)

def get_inventaire_stats() -> dict:
    """Obtenir les statistiques de l'inventaire"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_entries,
                COALESCE(SUM(total_purchase_cost), 0) as total_value,
                COALESCE(SUM(transport_cost), 0) as total_transport
            FROM inventaire
            WHERE is_depleted = 0
        """)
        
        result = cursor.fetchone()
        cursor.close()
        
        return {
            "total_entries": result[0],
            "total_value": float(result[1]),
            "total_transport": float(result[2])
        }

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="API Gestion Céréales",
    description="API pour la gestion des ventes, dépenses et inventaire de céréales",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ROUTES - VENTES
# ============================================================================
@app.post("/api/ventes", response_model=VenteResponse, status_code=201, tags=["Ventes"])
async def create_vente_route(vente: VenteCreate):
    """Créer une nouvelle vente"""
    try:
        result = create_vente(vente)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ventes", response_model=List[VenteResponse], tags=["Ventes"])
async def get_ventes_route(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    product_type: Optional[str] = None
):
    """Obtenir la liste des ventes avec filtres"""
    try:
        results = get_ventes(skip, limit, start_date, end_date, product_type)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ventes/{vente_id}", response_model=VenteResponse, tags=["Ventes"])
async def get_vente_route(vente_id: int):
    """Obtenir une vente par ID"""
    result = get_vente_by_id(vente_id)
    if not result:
        raise HTTPException(status_code=404, detail="Vente non trouvée")
    return result


@app.get("/api/ventes/stats/daily", tags=["Ventes"])
async def get_daily_stats_route(date: Optional[datetime] = None):
    """Obtenir les statistiques du jour"""
    if date is None:
        date = datetime.now()
    
    stats = get_daily_stats(date)
    
    # CORRECTION: Convertir tous les Decimal en float
    return convert_decimal_to_float(stats)
# ============================================================================
# ROUTES - DÉPENSES
# ============================================================================
@app.post("/api/depenses", response_model=DepenseResponse, status_code=201, tags=["Dépenses"])
async def create_depense_route(depense: DepenseCreate):
    """Créer une nouvelle dépense"""
    try:
        result = create_depense(depense)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/depenses", response_model=List[DepenseResponse], tags=["Dépenses"])
async def get_depenses_route(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    category: Optional[str] = None
):
    """Obtenir la liste des dépenses avec filtres"""
    try:
        results = get_depenses(skip, limit, start_date, end_date, category)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/depenses/stats/period", tags=["Dépenses"])
async def get_period_expenses_route(days: int = Query(1, ge=1, le=365)):
    """Obtenir les statistiques des dépenses sur une période"""
    return get_period_expenses(days)

# ============================================================================
# ROUTES - INVENTAIRE
# ============================================================================
@app.post("/api/inventaire", response_model=InventaireResponse, status_code=201, tags=["Inventaire"])
async def create_inventaire_route(inventaire: InventaireCreate):
    """Créer une nouvelle entrée d'inventaire"""
    try:
        result = create_inventaire(inventaire)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/inventaire", response_model=List[InventaireResponse], tags=["Inventaire"])
async def get_inventaires_route(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    cereal_type: Optional[str] = None,
    in_stock_only: bool = False
):
    """Obtenir la liste des inventaires avec filtres"""
    try:
        results = get_inventaires(skip, limit, cereal_type, in_stock_only)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.patch("/api/inventaire/{inventory_id}/stock", response_model=InventaireResponse, tags=["Inventaire"])
async def update_stock_route(
    inventory_id: int,
    quantity_sold: int = Query(..., gt=0)
):
    """Mettre à jour le stock après une vente"""
    try:
        result = update_stock(inventory_id, quantity_sold)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/inventaire/stats/summary", tags=["Inventaire"])
async def get_inventaire_stats_route():
    """Obtenir les statistiques de l'inventaire"""
    return get_inventaire_stats()

# ============================================================================
# ROUTES - GÉNÉRAL
# ============================================================================
@app.get("/", tags=["Général"])
async def root():
    """Route racine"""
    return {
        "message": "API Gestion Céréales",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ventes": "/api/ventes",
            "depenses": "/api/depenses",
            "inventaire": "/api/inventaire"
        }
    }

@app.get("/health", tags=["Général"])
async def health():
    """Vérification de santé de l'API"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

# ============================================================================
# ÉVÉNEMENTS DE DÉMARRAGE ET ARRÊT
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """Événement de démarrage de l'application"""
    print("🚀 Démarrage de l'API...")
    try:
        init_db_pool()
        print("✅ Pool de connexions initialisé")
        
        create_tables()
        print("✅ Tables créées/vérifiées")
        
        print("✅ API prête sur http://localhost:8000")
        print("📚 Documentation sur http://localhost:8000/docs")
    except Exception as e:
        print(f"❌ Erreur au démarrage: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt de l'application"""
    print("🛑 Arrêt de l'API...")
    if db_pool:
        db_pool.closeall()
        print("✅ Pool de connexions fermé")






# ============================================================================
# AJOUTER CES IMPORTS AU DÉBUT DU FICHIER main.py (après les imports existants)
# ============================================================================
from datetime import timedelta
from collections import defaultdict
import statistics
from decimal import Decimal

# ============================================================================
# FONCTION UTILITAIRE POUR CONVERSION DECIMAL
# ============================================================================
def convert_decimal_to_float(data):
    """Convertit tous les Decimal en float dans un dict ou une liste"""
    if isinstance(data, dict):
        return {k: convert_decimal_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_decimal_to_float(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data

# ============================================================================
# SERVICES ANALYTIQUES - Ajouter après les services existants
# ============================================================================

# --- Service d'analyse des ventes ---
def analyze_sales(db, start_date: datetime, end_date: datetime) -> dict:
    """Analyse détaillée des ventes sur une période"""
    cursor = db.cursor(cursor_factory=RealDictCursor)
    
    # Statistiques globales
    cursor.execute("""
        SELECT 
            COUNT(*) as total_transactions,
            COUNT(DISTINCT customer_name) as unique_customers,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as average_sale,
            SUM(quantity) as total_quantity,
            SUM(discount_amount) as total_discounts
        FROM ventes
        WHERE sale_date >= %s AND sale_date <= %s
    """, (start_date, end_date))
    
    stats = cursor.fetchone()
    
    # Ventes par produit
    cursor.execute("""
        SELECT 
            product_type,
            COUNT(*) as transaction_count,
            SUM(quantity) as total_quantity,
            SUM(total_amount) as total_revenue,
            AVG(unit_price) as avg_price
        FROM ventes
        WHERE sale_date >= %s AND sale_date <= %s
        GROUP BY product_type
        ORDER BY total_revenue DESC
    """, (start_date, end_date))
    
    by_product = cursor.fetchall()
    
    # Ventes par mode de paiement
    cursor.execute("""
        SELECT 
            payment_method,
            COUNT(*) as count,
            SUM(total_amount) as amount
        FROM ventes
        WHERE sale_date >= %s AND sale_date <= %s
        GROUP BY payment_method
    """, (start_date, end_date))
    
    by_payment = cursor.fetchall()
    
    # Ventes par jour de la semaine
    cursor.execute("""
        SELECT 
            EXTRACT(DOW FROM sale_date) as day_of_week,
            COUNT(*) as transaction_count,
            SUM(total_amount) as total_amount
        FROM ventes
        WHERE sale_date >= %s AND sale_date <= %s
        GROUP BY day_of_week
        ORDER BY day_of_week
    """, (start_date, end_date))
    
    by_weekday = cursor.fetchall()
    
    cursor.close()
    
    return convert_decimal_to_float({
        "global_stats": dict(stats) if stats else {},
        "by_product": [dict(r) for r in by_product],
        "by_payment": [dict(r) for r in by_payment],
        "by_weekday": [dict(r) for r in by_weekday]
    })

# --- Service d'analyse des stocks ---
def analyze_inventory(db) -> dict:
    """Analyse détaillée de l'inventaire"""
    cursor = db.cursor(cursor_factory=RealDictCursor)
    
    # Statistiques par produit
    cursor.execute("""
        SELECT 
            cereal_type,
            SUM(current_stock) as total_bowls,
            SUM(sac_quantity) as total_sacs,
            AVG(price_per_bowl) as avg_purchase_price,
            SUM(current_stock * price_per_bowl) as stock_value,
            COUNT(*) as entry_count
        FROM inventaire
        WHERE is_depleted = 0
        GROUP BY cereal_type
        ORDER BY stock_value DESC
    """)
    
    by_product = cursor.fetchall()
    
    # Produits en stock faible
    cursor.execute("""
        SELECT 
            cereal_type,
            SUM(current_stock) as total_stock
        FROM inventaire
        WHERE is_depleted = 0
        GROUP BY cereal_type
        HAVING SUM(current_stock) < 200
    """)
    
    low_stock = cursor.fetchall()
    
    # Valeur totale du stock
    cursor.execute("""
        SELECT 
            SUM(current_stock) as total_bowls,
            SUM(current_stock * price_per_bowl) as total_value,
            SUM(transport_cost) as total_transport
        FROM inventaire
        WHERE is_depleted = 0
    """)
    
    totals = cursor.fetchone()
    
    cursor.close()
    
    return convert_decimal_to_float({
        "by_product": [dict(r) for r in by_product],
        "low_stock_items": [dict(r) for r in low_stock],
        "totals": dict(totals) if totals else {}
    })

# --- Service d'analyse des dépenses ---
def analyze_expenses(db, start_date: datetime, end_date: datetime) -> dict:
    """Analyse détaillée des dépenses"""
    cursor = db.cursor(cursor_factory=RealDictCursor)
    
    # Dépenses par catégorie
    cursor.execute("""
        SELECT 
            expense_category,
            COUNT(*) as count,
            SUM(expense_amount) as total_amount,
            AVG(expense_amount) as avg_amount
        FROM depenses
        WHERE expense_date >= %s AND expense_date <= %s
        GROUP BY expense_category
        ORDER BY total_amount DESC
    """, (start_date, end_date))
    
    by_category = cursor.fetchall()
    
    # Total des dépenses
    cursor.execute("""
        SELECT 
            COUNT(*) as total_count,
            SUM(expense_amount) as total_expenses,
            AVG(expense_amount) as avg_expense
        FROM depenses
        WHERE expense_date >= %s AND expense_date <= %s
    """, (start_date, end_date))
    
    totals = cursor.fetchone()
    
    cursor.close()
    
    return convert_decimal_to_float({
        "by_category": [dict(r) for r in by_category],
        "totals": dict(totals) if totals else {}
    })

# --- Calcul du taux de rotation ---
def calculate_rotation_rate(db) -> float:
    """Calcule le taux de rotation des stocks en jours"""
    cursor = db.cursor()
    
    cursor.execute("SELECT AVG(current_stock) FROM inventaire WHERE is_depleted = 0")
    avg_stock = cursor.fetchone()[0] or 0
    avg_stock = float(avg_stock) if avg_stock else 0
    
    cursor.execute("""
        SELECT AVG(daily_sales) FROM (
            SELECT DATE(sale_date) as date, SUM(quantity) as daily_sales
            FROM ventes
            WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(sale_date)
        ) as daily
    """)
    avg_daily_sales = cursor.fetchone()[0] or 1
    avg_daily_sales = float(avg_daily_sales) if avg_daily_sales else 1
    
    cursor.close()
    
    if avg_daily_sales > 0:
        return round(avg_stock / avg_daily_sales, 1)
    return 0.0

# --- Calcul des tendances ---
def calculate_trends(db) -> dict:
    """Calcule les tendances pour toutes les métriques"""
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN sale_date >= CURRENT_DATE - INTERVAL '7 days' 
                THEN total_amount ELSE 0 END) as current_week,
            SUM(CASE WHEN sale_date >= CURRENT_DATE - INTERVAL '14 days' 
                AND sale_date < CURRENT_DATE - INTERVAL '7 days'
                THEN total_amount ELSE 0 END) as previous_week
        FROM ventes
    """)
    
    sales_trend = cursor.fetchone()
    
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN expense_date >= CURRENT_DATE - INTERVAL '7 days' 
                THEN expense_amount ELSE 0 END) as current_week,
            SUM(CASE WHEN expense_date >= CURRENT_DATE - INTERVAL '14 days' 
                AND expense_date < CURRENT_DATE - INTERVAL '7 days'
                THEN expense_amount ELSE 0 END) as previous_week
        FROM depenses
    """)
    
    expenses_trend = cursor.fetchone()
    
    cursor.execute("SELECT SUM(current_stock) FROM inventaire WHERE is_depleted = 0")
    current_stock = cursor.fetchone()[0] or 0
    current_stock = float(current_stock) if current_stock else 0
    
    cursor.close()
    
    def calc_percentage_change(current, previous):
        current = float(current) if current else 0
        previous = float(previous) if previous else 0
        if previous > 0:
            return round(((current - previous) / previous) * 100, 1)
        return 0.0
    
    sales_current = float(sales_trend[0]) if sales_trend[0] else 0
    sales_previous = float(sales_trend[1]) if sales_trend[1] else 0
    
    expenses_current = float(expenses_trend[0]) if expenses_trend[0] else 0
    expenses_previous = float(expenses_trend[1]) if expenses_trend[1] else 0
    
    return {
        "sales": {
            "current": sales_current,
            "change": calc_percentage_change(sales_current, sales_previous),
            "is_increasing": sales_current > sales_previous
        },
        "expenses": {
            "current": expenses_current,
            "change": calc_percentage_change(expenses_current, expenses_previous),
            "is_increasing": expenses_current > expenses_previous
        },
        "stock": {
            "current": current_stock,
            "change": 8.0,
            "is_increasing": True
        }
    }

# --- Recommandations de prix ---
def generate_price_recommendations(db) -> List[dict]:
    """Génère des recommandations de prix basées sur les données"""
    cursor = db.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        WITH product_prices AS (
            SELECT 
                i.cereal_type,
                AVG(i.price_per_bowl) as avg_purchase_price,
                AVG(v.unit_price) as avg_selling_price,
                SUM(i.current_stock) as current_stock
            FROM inventaire i
            LEFT JOIN ventes v ON v.product_type = i.cereal_type
            WHERE i.is_depleted = 0
            GROUP BY i.cereal_type
        )
        SELECT * FROM product_prices
    """)
    
    products = cursor.fetchall()
    cursor.close()
    
    recommendations = []
    
    for product in products:
        purchase_price = float(product['avg_purchase_price'] or 0)
        selling_price = float(product['avg_selling_price'] or purchase_price * 1.5)
        stock = int(product['current_stock'] or 0)
        
        if purchase_price > 0:
            current_margin = ((selling_price - purchase_price) / purchase_price) * 100
            
            if current_margin < 30:
                recommended_min = purchase_price * 1.35
                recommended_max = purchase_price * 1.50
                reason = "Marge actuelle trop faible, augmentation recommandée"
            elif current_margin > 50:
                recommended_min = purchase_price * 1.40
                recommended_max = purchase_price * 1.45
                reason = "Marge élevée, ajustement pour compétitivité"
            elif stock < 200:
                recommended_min = selling_price * 1.05
                recommended_max = selling_price * 1.15
                reason = "Stock faible, augmentation pour gérer la demande"
            else:
                recommended_min = selling_price * 0.95
                recommended_max = selling_price * 1.05
                reason = "Prix optimal actuel, maintenir la fourchette"
            
            recommendations.append({
                "product_name": product['cereal_type'],
                "purchase_price": round(purchase_price, 0),
                "current_price": round(selling_price, 0),
                "current_margin": round(current_margin, 1),
                "recommended_min": round(recommended_min, 0),
                "recommended_max": round(recommended_max, 0),
                "recommendation_reason": reason
            })
    
    return recommendations

# --- Prédictions ---
def generate_predictions(db) -> dict:
    """Génère des prédictions pour les 30 prochains jours"""
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT DATE(sale_date) as date, SUM(total_amount) as daily_revenue
        FROM ventes
        WHERE sale_date >= CURRENT_DATE - INTERVAL '60 days'
        GROUP BY DATE(sale_date)
        ORDER BY date
    """)
    
    daily_revenues = cursor.fetchall()
    cursor.close()
    
    if len(daily_revenues) > 7:
        revenues = [float(r[1]) for r in daily_revenues]
        avg_revenue = statistics.mean(revenues[-7:])
        trend = statistics.mean(revenues[-7:]) - statistics.mean(revenues[-14:-7]) if len(revenues) >= 14 else 0
        
        predictions = []
        for day in range(30):
            predicted_revenue = avg_revenue + (trend * (day / 7))
            predictions.append(max(0, predicted_revenue))
        
        total_predicted = sum(predictions)
    else:
        avg_revenue = 0
        predictions = [0] * 30
        total_predicted = 0
    
    return {
        "next_30_days": {
            "predicted_revenue": round(total_predicted, 0),
            "daily_average": round(avg_revenue, 0),
            "daily_predictions": [round(p, 0) for p in predictions]
        },
        "scenarios": {
            "optimistic": round(total_predicted * 1.2, 0),
            "realistic": round(total_predicted, 0),
            "pessimistic": round(total_predicted * 0.8, 0)
        }
    }

# --- Insights ---
def generate_insights(db) -> dict:
    """Génère des insights basés sur les données"""
    insights = {
        "alerts": [],
        "opportunities": [],
        "warnings": [],
        "achievements": []
    }
    
    inventory_analysis = analyze_inventory(db)
    
    if inventory_analysis["low_stock_items"]:
        for item in inventory_analysis["low_stock_items"]:
            insights["alerts"].append({
                "type": "stock_alert",
                "message": f"Stock faible pour {item['cereal_type']}: {item['total_stock']} bols restants",
                "priority": "high"
            })
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    sales_analysis = analyze_sales(db, start_date, end_date)
    
    if sales_analysis["global_stats"]:
        # total_revenue = float(sales_analysis["global_stats"].get("total_revenue", 0))
        # Vérifier que global_stats existe et n'est pas None
        global_stats = sales_analysis.get("global_stats") or {}
        # total_revenue = float(global_stats.get("total_revenue", 0))
        total_revenue = float(global_stats.get("total_revenue") or 0)
        
        if total_revenue > 500000:
            insights["achievements"].append({
                "type": "sales_milestone",
                "message": f"Excellent! Revenus de {int(total_revenue):,} FCFA cette semaine",
                "icon": "🎉"
            })
        
        if sales_analysis["by_product"]:
            top_product = sales_analysis["by_product"][0]
            insights["opportunities"].append({
                "type": "trending_product",
                "message": f"{top_product['product_type'].capitalize()} est très demandé!",
                "icon": "📈"
            })
    
    recommendations = generate_price_recommendations(db)
    for rec in recommendations:
        if rec["current_margin"] < 25:
            insights["warnings"].append({
                "type": "low_margin",
                "message": f"Marge faible sur {rec['product_name']}: {rec['current_margin']}%",
                "priority": "medium"
            })
    
    return insights

# ============================================================================
# ROUTES API DASHBOARD - Utilisant @app.get()
# ============================================================================

@app.get("/api/dashboard/stats", tags=["Dashboard"])
async def get_dashboard_stats(period: str = "month"):
    """Obtenir toutes les statistiques du dashboard"""
    
    end_date = datetime.now()
    
    if period == "today":
        start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start_date = end_date - timedelta(days=7)
    elif period == "month" or period == "30days":
        start_date = end_date - timedelta(days=30)
    elif period == "6months":
        start_date = end_date - timedelta(days=180)
    elif period == "year":
        start_date = end_date - timedelta(days=365)
    else:
        start_date = end_date - timedelta(days=30)
    
    with get_db_connection() as db:
        sales_analysis = analyze_sales(db, start_date, end_date)
        inventory_analysis = analyze_inventory(db)
        expenses_analysis = analyze_expenses(db, start_date, end_date)
        trends = calculate_trends(db)
        rotation_rate = calculate_rotation_rate(db)
        recommendations = generate_price_recommendations(db)
        predictions = generate_predictions(db)
        insights = generate_insights(db)
        
        # Vérifier que global_stats existe et n'est pas None
        global_stats = sales_analysis.get("global_stats") or {}
        # Utiliser or 0 pour gérer le cas où la valeur est None
        total_revenue = float(global_stats.get("total_revenue") or 0)
        
        # Vérifier que totals existe et n'est pas None
        expense_totals = expenses_analysis.get("totals") or {}
        total_expenses = float(expense_totals.get("total_expenses") or 0)
        
        net_profit = total_revenue - total_expenses
        profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Vérifier que inventory_analysis a les bonnes clés
        inventory_totals = inventory_analysis.get("totals") or {}
        inventory_by_product = inventory_analysis.get("by_product") or []
        low_stock_items = inventory_analysis.get("low_stock_items") or []
        
        main_stats = {
            "total_stock_bowls": int(inventory_totals.get("total_bowls") or 0),
            "total_stock_sacs": sum([int(p.get("total_sacs") or 0) for p in inventory_by_product]),
            "total_sales": round(total_revenue, 0),
            "total_expenses": round(total_expenses, 0),
            "net_profit": round(net_profit, 0),
            "profit_margin": round(profit_margin, 1),
            "total_customers": int(global_stats.get("unique_customers") or 0),
            "total_transactions": int(global_stats.get("total_transactions") or 0),
            "total_arrivals": len(inventory_by_product),
            "rotation_rate": rotation_rate or 0,
            "low_stock_items": len(low_stock_items)
        }
        
        # Calculer les avances et crédits
        cursor = db.cursor(cursor_factory=RealDictCursor)

        # Total des avances données
        cursor.execute("""
            SELECT 
                SUM(advance_amount) as total_advances,
                SUM(total_amount - advance_amount) as total_remaining
            FROM ventes
            WHERE payment_method = 'advance'
            AND sale_date >= %s AND sale_date <= %s
        """, (start_date, end_date))
        advances_data = cursor.fetchone()

        # Total des ventes à crédit
        cursor.execute("""
            SELECT 
                COUNT(*) as credit_count,
                SUM(total_amount) as total_credit
            FROM ventes
            WHERE payment_method = 'credit'
            AND sale_date >= %s AND sale_date <= %s
        """, (start_date, end_date))
        credit_data = cursor.fetchone()

        cursor.close()

        # Ajouter aux stats principales (avec double vérification de None)
        main_stats["total_advances"] = round(float((advances_data.get("total_advances") if advances_data else None) or 0), 0)
        main_stats["total_remaining"] = round(float((advances_data.get("total_remaining") if advances_data else None) or 0), 0)
        main_stats["credit_count"] = int((credit_data.get("credit_count") if credit_data else None) or 0)
        main_stats["total_credit"] = round(float((credit_data.get("total_credit") if credit_data else None) or 0), 0)
        
        return {
            "stats": main_stats,
            "trends": trends,
            "sales_analysis": sales_analysis,
            "inventory_analysis": inventory_analysis,
            "expenses_analysis": expenses_analysis,
            "recommendations": recommendations,
            "predictions": predictions,
            "insights": insights
        }


@app.get("/api/dashboard/charts/sales-vs-purchases", tags=["Dashboard"])
async def get_sales_purchases_chart(months: int = 6):
    """Données pour le graphique Ventes vs Achats"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                TO_CHAR(sale_date, 'Mon') as month,
                EXTRACT(MONTH FROM sale_date) as month_num,
                SUM(total_amount) as total
            FROM ventes
            WHERE sale_date >= CURRENT_DATE - INTERVAL '%s months'
            GROUP BY month, month_num
            ORDER BY month_num
        """, (months,))
        
        sales_data = cursor.fetchall()
        
        cursor.execute("""
            SELECT 
                TO_CHAR(expense_date, 'Mon') as month,
                EXTRACT(MONTH FROM expense_date) as month_num,
                SUM(expense_amount) as total
            FROM depenses
            WHERE expense_date >= CURRENT_DATE - INTERVAL '%s months'
            GROUP BY month, month_num
            ORDER BY month_num
        """, (months,))
        
        expenses_data = cursor.fetchall()
        cursor.close()
        
        months_fr = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        
        sales_by_month = {r['month']: float(r['total']) for r in sales_data}
        expenses_by_month = {r['month']: float(r['total']) for r in expenses_data}
        
        labels = list(set([r['month'] for r in sales_data] + [r['month'] for r in expenses_data]))
        labels.sort(key=lambda x: months_fr.index(x) if x in months_fr else 0)
        
        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Achats",
                    "data": [expenses_by_month.get(m, 0) for m in labels]
                },
                {
                    "label": "Ventes",
                    "data": [sales_by_month.get(m, 0) for m in labels]
                }
            ]
        }

@app.get("/api/dashboard/charts/stock-distribution", tags=["Dashboard"])
async def get_stock_distribution_chart():
    """Données pour le graphique de répartition des stocks"""
    with get_db_connection() as db:
        inventory_analysis = analyze_inventory(db)
        
        labels = [p["cereal_type"].capitalize() for p in inventory_analysis["by_product"]]
        data = [int(p["total_bowls"]) for p in inventory_analysis["by_product"]]
        
        return {
            "labels": labels,
            "datasets": [{
                "data": data,
                "backgroundColor": [
                    'rgba(22, 163, 74, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(59, 130, 246, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(147, 51, 234, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(156, 163, 175, 0.8)'
                ]
            }]
        }

@app.get("/api/dashboard/charts/revenue-profit", tags=["Dashboard"])
async def get_revenue_profit_chart():
    """Données pour le graphique Revenus et Bénéfices"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            WITH monthly_data AS (
                SELECT 
                    TO_CHAR(sale_date, 'Mon') as month,
                    EXTRACT(MONTH FROM sale_date) as month_num,
                    SUM(total_amount) as revenue,
                    SUM(total_amount - (quantity * unit_price * 0.7)) as profit
                FROM ventes
                WHERE sale_date >= CURRENT_DATE - INTERVAL '12 months'
                GROUP BY month, month_num
                ORDER BY month_num
            )
            SELECT * FROM monthly_data
        """)
        
        monthly_data = cursor.fetchall()
        cursor.close()
        
        labels = [r['month'] for r in monthly_data]
        revenues = [float(r['revenue']) for r in monthly_data]
        profits = [float(r['profit']) for r in monthly_data]
        
        return {
            "labels": labels,
            "datasets": [
                {"label": "Revenus", "data": revenues},
                {"label": "Bénéfices", "data": profits}
            ]
        }

@app.get("/api/dashboard/charts/product-sales", tags=["Dashboard"])
async def get_product_sales_chart():
    """Données pour les quantités vendues par produit"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT product_type, SUM(quantity) as total_quantity
            FROM ventes
            WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY product_type
            ORDER BY total_quantity DESC
            LIMIT 6
        """)
        
        products = cursor.fetchall()
        cursor.close()
        
        return {
            "labels": [p['product_type'].capitalize() for p in products],
            "datasets": [{
                "label": "Bols vendus",
                "data": [int(p['total_quantity']) for p in products],
                "backgroundColor": [
                    'rgba(22, 163, 74, 0.7)',
                    'rgba(245, 158, 11, 0.7)',
                    'rgba(59, 130, 246, 0.7)',
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(147, 51, 234, 0.7)',
                    'rgba(16, 185, 129, 0.7)'
                ]
            }]
        }

@app.get("/api/dashboard/charts/weekday-sales", tags=["Dashboard"])
async def get_weekday_sales_chart():
    """Données pour les ventes par jour de la semaine"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT EXTRACT(DOW FROM sale_date) as day_num, COUNT(*) as transaction_count
            FROM ventes
            WHERE sale_date >= CURRENT_DATE - INTERVAL '60 days'
            GROUP BY day_num
            ORDER BY day_num
        """)
        
        weekdays_data = cursor.fetchall()
        cursor.close()
        
        days_fr = ['Dimanche', 'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi']
        day_data = {int(r['day_num']): int(r['transaction_count']) for r in weekdays_data}
        max_count = max(day_data.values()) if day_data else 1
        normalized_data = [(day_data.get(i, 0) / max_count * 100) for i in range(7)]
        
        return {
            "labels": days_fr,
            "datasets": [{
                "label": "Ventes moyennes",
                "data": normalized_data,
                "backgroundColor": "rgba(22, 163, 74, 0.2)",
                "borderColor": "rgba(22, 163, 74, 1)"
            }]
        }

@app.get("/api/dashboard/stock-details", tags=["Dashboard"])
async def get_stock_details():
    """Obtenir les détails du stock par céréale"""
    with get_db_connection() as db:
        inventory_analysis = analyze_inventory(db)
        
        stock_details = []
        for product in inventory_analysis["by_product"]:
            total_bowls = int(product["total_bowls"])
            
            if total_bowls > 500:
                status, status_class = "Bon", "success"
            elif total_bowls > 200:
                status, status_class = "Moyen", "warning"
            else:
                status, status_class = "Faible", "danger"
            
            stock_details.append({
                "cereal_type": product["cereal_type"].capitalize(),
                "quantity_bowls": total_bowls,
                "quantity_sacs": int(product["total_sacs"]),
                "purchase_price": round(float(product["avg_purchase_price"]), 0),
                "stock_value": round(float(product["stock_value"]), 0),
                "status": status,
                "status_class": status_class
            })
        
        return stock_details

@app.get("/api/dashboard/recommendations", tags=["Dashboard"])
async def get_recommendations():
    """Obtenir les recommandations de prix"""
    with get_db_connection() as db:
        return generate_price_recommendations(db)

@app.get("/api/dashboard/insights", tags=["Dashboard"])
async def get_insights():
    """Obtenir les insights et alertes"""
    with get_db_connection() as db:
        return generate_insights(db)

@app.get("/api/dashboard/predictions", tags=["Dashboard"])
async def get_predictions():
    """Obtenir les prédictions pour les 30 prochains jours"""
    with get_db_connection() as db:
        return generate_predictions(db)
















# ============================================================================
# ROUTES API HISTORIQUE - PostgreSQL Version
# À copier-coller dans votre fichier main.py après les routes dashboard
# ============================================================================

from typing import Optional, List
from datetime import date, datetime, timedelta
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================

class HistoriqueStats(BaseModel):
    caisse_actuelle: float
    total_ventes: float
    total_depenses: float
    total_avances: float
    reste_a_payer: float
    total_credits: float

class ProductRanking(BaseModel):
    product_type: str
    total_quantity: float
    total_sales: int
    total_revenue: float

class VenteAvance(BaseModel):
    id: int
    receipt_number: str
    sale_date: datetime  # ✅ Changé de 'date' à 'datetime'
    sale_time: str
    customer_name: Optional[str]
    customer_phone: Optional[str]
    product_type: str
    quantity: int  # ✅ Changé de 'float' à 'int'
    unit_type: str
    total_amount: float
    payment_method: str
    advance_amount: Optional[float]
    balance_due: Optional[float]  # ✅ Déjà optionnel, c'est bon


class VenteCredit(BaseModel):
    id: int
    receipt_number: str
    sale_date: date
    sale_time: str
    customer_name: Optional[str]
    customer_phone: Optional[str]
    product_type: str
    quantity: float
    unit_type: str
    total_amount: float
    payment_method: str

class TopClient(BaseModel):
    customer_name: str
    customer_phone: Optional[str]
    nombre_achats: int
    quantite_totale: float
    montant_total: float


@app.get("/api/historique/stats", response_model=HistoriqueStats, tags=["Historique"])
async def get_historique_stats():
    """
    Récupère les statistiques globales pour l'historique
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            # ✅ CORRECTION : Calculer la caisse à partir de ventes - dépenses
            cursor.execute("""
                SELECT 
                    COALESCE(
                        (SELECT SUM(total_amount) FROM ventes WHERE payment_method IN ('cash', 'mobile', 'bank')) 
                        - 
                        (SELECT COALESCE(SUM(expense_amount), 0) FROM depenses),
                        0
                    ) as caisse_actuelle
            """)
            caisse_result = cursor.fetchone()
            caisse_actuelle = float(caisse_result['caisse_actuelle']) if caisse_result else 0.0
            
            # Total des ventes
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as total 
                FROM ventes
            """)
            total_ventes = float(cursor.fetchone()['total'])
            
            # Total des dépenses
            cursor.execute("""
                SELECT COALESCE(SUM(expense_amount), 0) as total 
                FROM depenses
            """)
            total_depenses = float(cursor.fetchone()['total'])
            
            # Total des avances données
            cursor.execute("""
                SELECT COALESCE(SUM(advance_amount), 0) as total
                FROM ventes 
                WHERE payment_method = 'advance'
            """)
            total_avances = float(cursor.fetchone()['total'])
            
            # Reste à payer (avances non complétées)
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount - COALESCE(advance_amount, 0)), 0) as total
                FROM ventes 
                WHERE payment_method = 'advance' 
                AND (total_amount - COALESCE(advance_amount, 0)) > 0
            """)
            reste_a_payer = float(cursor.fetchone()['total'])
            
            # Total des crédits en cours
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as total
                FROM ventes 
                WHERE payment_method = 'credit'
            """)
            total_credits = float(cursor.fetchone()['total'])
            
            cursor.close()
            
            return HistoriqueStats(
                caisse_actuelle=caisse_actuelle,
                total_ventes=total_ventes,
                total_depenses=total_depenses,
                total_avances=total_avances,
                reste_a_payer=reste_a_payer,
                total_credits=total_credits
            )
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des stats: {str(e)}")

# ============================================================================
# ROUTE 2: VENTES AVEC AVANCES
# ============================================================================

@app.get("/api/historique/avances", tags=["Historique"])
async def get_ventes_avances():
    """
    Récupère toutes les ventes payées avec avance
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    id,
                    receipt_number,
                    sale_date,
                    sale_time,
                    customer_name,
                    customer_phone,
                    product_type,
                    quantity,
                    unit_type,
                    total_amount,
                    payment_method,
                    advance_amount,
                    (total_amount - COALESCE(advance_amount, 0)) as balance_due  -- ✅ Calculé
                FROM ventes
                WHERE payment_method = 'advance'
                ORDER BY sale_date DESC, sale_time DESC
            """)
            
            avances = cursor.fetchall()
            cursor.close()
            
            return [convert_decimal_to_float(dict(row)) for row in avances]
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur avances: {str(e)}")

# ============================================================================
# ROUTE 3: VENTES À CRÉDIT
# ============================================================================

@app.get("/api/historique/credits", response_model=List[VenteCredit], tags=["Historique"])
async def get_ventes_credits():
    """
    Récupère toutes les ventes à crédit
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    id,
                    receipt_number,
                    sale_date,
                    sale_time,
                    customer_name,
                    customer_phone,
                    product_type,
                    quantity,
                    unit_type,
                    total_amount,
                    payment_method
                FROM ventes
                WHERE payment_method = 'credit'
                ORDER BY sale_date DESC, sale_time DESC
            """)
            
            credits = cursor.fetchall()
            cursor.close()
            
            return [VenteCredit(**convert_decimal_to_float(dict(row))) for row in credits]
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des crédits: {str(e)}")

# ============================================================================
# ROUTE 4: CLASSEMENT DES PRODUITS
# ============================================================================

@app.get("/api/historique/classement", response_model=List[ProductRanking], tags=["Historique"])
async def get_product_ranking():
    """
    Récupère le classement des produits par ventes
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    product_type,
                    SUM(quantity) as total_quantity,
                    COUNT(*) as total_sales,
                    SUM(total_amount) as total_revenue
                FROM ventes
                GROUP BY product_type
                ORDER BY total_revenue DESC
            """)
            
            classement = cursor.fetchall()
            cursor.close()
            
            return [ProductRanking(**convert_decimal_to_float(dict(row))) for row in classement]
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du classement: {str(e)}")

# ============================================================================
# ROUTE 5: RAPPORT DÉTAILLÉ PAR PÉRIODE
# ============================================================================

@app.get("/api/historique/rapport", tags=["Historique"])
async def get_rapport_periode(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """
    Génère un rapport détaillé pour une période donnée
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Dates par défaut: 30 derniers jours
            if not start_date:
                start_date = date.today() - timedelta(days=30)
            if not end_date:
                end_date = date.today()
            
            # Ventes de la période
            cursor.execute("""
                SELECT 
                    COUNT(*) as nombre_ventes,
                    SUM(quantity) as quantite_totale,
                    SUM(total_amount) as montant_total
                FROM ventes
                WHERE sale_date BETWEEN %s AND %s
            """, (start_date, end_date))
            ventes_stats = cursor.fetchone()
            
            # Dépenses de la période
            cursor.execute("""
                SELECT 
                    COUNT(*) as nombre_depenses,
                    SUM(expense_amount) as montant_total
                FROM depenses
                WHERE expense_date BETWEEN %s AND %s
            """, (start_date, end_date))
            depenses_stats = cursor.fetchone()
            
            # Ventes par produit
            cursor.execute("""
                SELECT 
                    product_type,
                    COUNT(*) as nombre,
                    SUM(quantity) as quantite,
                    SUM(total_amount) as montant
                FROM ventes
                WHERE sale_date BETWEEN %s AND %s
                GROUP BY product_type
                ORDER BY montant DESC
            """, (start_date, end_date))
            ventes_par_produit = cursor.fetchall()
            
            # Dépenses par catégorie
            cursor.execute("""
                SELECT 
                    expense_category,
                    COUNT(*) as nombre,
                    SUM(expense_amount) as montant
                FROM depenses
                WHERE expense_date BETWEEN %s AND %s
                GROUP BY expense_category
                ORDER BY montant DESC
            """, (start_date, end_date))
            depenses_par_categorie = cursor.fetchall()
            
            cursor.close()
            
            return convert_decimal_to_float({
                'periode': {
                    'debut': str(start_date),
                    'fin': str(end_date)
                },
                'ventes': {
                    'nombre': int(ventes_stats['nombre_ventes'] or 0),
                    'quantite_totale': float(ventes_stats['quantite_totale'] or 0),
                    'montant_total': float(ventes_stats['montant_total'] or 0),
                    'par_produit': [dict(row) for row in ventes_par_produit]
                },
                'depenses': {
                    'nombre': int(depenses_stats['nombre_depenses'] or 0),
                    'montant_total': float(depenses_stats['montant_total'] or 0),
                    'par_categorie': [dict(row) for row in depenses_par_categorie]
                },
                'bilan': {
                    'entrees': float(ventes_stats['montant_total'] or 0),
                    'sorties': float(depenses_stats['montant_total'] or 0),
                    'solde': float(ventes_stats['montant_total'] or 0) - float(depenses_stats['montant_total'] or 0)
                }
            })
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du rapport: {str(e)}")

# ============================================================================
# ROUTE 6: VENTES PAR PÉRIODE (avec filtres avancés)
# ============================================================================

@app.get("/api/historique/ventes-periode", tags=["Historique"])
async def get_ventes_periode(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    product_type: Optional[str] = None,
    payment_method: Optional[str] = None
):
    """
    Récupère les ventes avec filtres avancés
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Construction de la requête dynamique
            query = """
                SELECT 
                    id,
                    receipt_number,
                    sale_date,
                    sale_time,
                    customer_name,
                    customer_phone,
                    product_type,
                    quantity,
                    unit_type,
                    unit_price,
                    total_amount,
                    payment_method,
                    advance_amount,
                    balance_due
                FROM ventes
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND sale_date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND sale_date <= %s"
                params.append(end_date)
            
            if product_type:
                query += " AND product_type = %s"
                params.append(product_type)
            
            if payment_method:
                query += " AND payment_method = %s"
                params.append(payment_method)
            
            query += " ORDER BY sale_date DESC, sale_time DESC"
            
            cursor.execute(query, params)
            ventes = cursor.fetchall()
            cursor.close()
            
            return [convert_decimal_to_float(dict(row)) for row in ventes]
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des ventes: {str(e)}")

# ============================================================================
# ROUTE 7: STATISTIQUES MENSUELLES
# ============================================================================

@app.get("/api/historique/stats-mensuelles", tags=["Historique"])
async def get_stats_mensuelles(year: Optional[int] = None):
    """
    Récupère les statistiques mensuelles pour une année
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            if not year:
                year = datetime.now().year
            
            # Ventes par mois
            cursor.execute("""
                SELECT 
                    EXTRACT(MONTH FROM sale_date)::integer as mois,
                    COUNT(*) as nombre_ventes,
                    SUM(quantity) as quantite_totale,
                    SUM(total_amount) as montant_total
                FROM ventes
                WHERE EXTRACT(YEAR FROM sale_date) = %s
                GROUP BY mois
                ORDER BY mois
            """, (year,))
            ventes_mensuelles = cursor.fetchall()
            
            # Dépenses par mois
            cursor.execute("""
                SELECT 
                    EXTRACT(MONTH FROM expense_date)::integer as mois,
                    COUNT(*) as nombre_depenses,
                    SUM(expense_amount) as montant_total
                FROM depenses
                WHERE EXTRACT(YEAR FROM expense_date) = %s
                GROUP BY mois
                ORDER BY mois
            """, (year,))
            depenses_mensuelles = cursor.fetchall()
            
            cursor.close()
            
            return convert_decimal_to_float({
                'annee': year,
                'ventes_mensuelles': [dict(row) for row in ventes_mensuelles],
                'depenses_mensuelles': [dict(row) for row in depenses_mensuelles]
            })
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des stats mensuelles: {str(e)}")

# ============================================================================
# ROUTE 8: TOP CLIENTS
# ============================================================================

@app.get("/api/historique/top-clients", response_model=List[TopClient], tags=["Historique"])
async def get_top_clients(limit: int = 10):
    """
    Récupère les meilleurs clients par montant d'achats
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    customer_name,
                    customer_phone,
                    COUNT(*) as nombre_achats,
                    SUM(quantity) as quantite_totale,
                    SUM(total_amount) as montant_total
                FROM ventes
                WHERE customer_name IS NOT NULL AND customer_name != ''
                GROUP BY customer_name, customer_phone
                ORDER BY montant_total DESC
                LIMIT %s
            """, (limit,))
            
            top_clients = cursor.fetchall()
            cursor.close()
            
            return [TopClient(**convert_decimal_to_float(dict(row))) for row in top_clients]
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des top clients: {str(e)}")

# ============================================================================
# ROUTE 9: EVOLUTION DES VENTES (Graphique tendances)
# ============================================================================

@app.get("/api/historique/evolution-ventes", tags=["Historique"])
async def get_evolution_ventes(days: int = 30):
    """
    Récupère l'évolution des ventes jour par jour
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            start_date = date.today() - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    sale_date,
                    COUNT(*) as nombre_transactions,
                    SUM(quantity) as quantite_totale,
                    SUM(total_amount) as montant_total
                FROM ventes
                WHERE sale_date >= %s
                GROUP BY sale_date
                ORDER BY sale_date ASC
            """, (start_date,))
            
            evolution = cursor.fetchall()
            cursor.close()
            
            return {
                'labels': [str(row['sale_date']) for row in evolution],
                'transactions': [int(row['nombre_transactions']) for row in evolution],
                'quantites': [float(row['quantite_totale']) for row in evolution],
                'montants': [float(row['montant_total']) for row in evolution]
            }
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération de l'évolution: {str(e)}")

# ============================================================================
# ROUTE 10: RÉSUMÉ RAPIDE POUR EXPORT PDF
# ============================================================================

@app.get("/api/historique/resume-export", tags=["Historique"])
async def get_resume_export(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):
    """
    Récupère un résumé complet pour l'export PDF
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            if not start_date:
                start_date = date.today() - timedelta(days=30)
            if not end_date:
                end_date = date.today()
            
            # Statistiques générales
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_ventes,
                    SUM(total_amount) as ca_total,
                    AVG(total_amount) as panier_moyen,
                    COUNT(DISTINCT customer_name) as nb_clients
                FROM ventes
                WHERE sale_date BETWEEN %s AND %s
            """, (start_date, end_date))
            stats_ventes = cursor.fetchone()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_depenses,
                    SUM(expense_amount) as montant_total
                FROM depenses
                WHERE expense_date BETWEEN %s AND %s
            """, (start_date, end_date))
            stats_depenses = cursor.fetchone()
            
            # Top 5 produits
            cursor.execute("""
                SELECT 
                    product_type,
                    SUM(quantity) as quantite,
                    SUM(total_amount) as montant
                FROM ventes
                WHERE sale_date BETWEEN %s AND %s
                GROUP BY product_type
                ORDER BY montant DESC
                LIMIT 5
            """, (start_date, end_date))
            top_produits = cursor.fetchall()
            
            # Répartition paiements
            cursor.execute("""
                SELECT 
                    payment_method,
                    COUNT(*) as nombre,
                    SUM(total_amount) as montant
                FROM ventes
                WHERE sale_date BETWEEN %s AND %s
                GROUP BY payment_method
            """, (start_date, end_date))
            repartition_paiements = cursor.fetchall()
            
            cursor.close()
            
            ca_total = float(stats_ventes['ca_total'] or 0)
            depenses_total = float(stats_depenses['montant_total'] or 0)
            benefice_net = ca_total - depenses_total
            marge = (benefice_net / ca_total * 100) if ca_total > 0 else 0
            
            return convert_decimal_to_float({
                'periode': {
                    'debut': str(start_date),
                    'fin': str(end_date)
                },
                'resume': {
                    'nombre_ventes': int(stats_ventes['total_ventes'] or 0),
                    'ca_total': ca_total,
                    'panier_moyen': float(stats_ventes['panier_moyen'] or 0),
                    'nombre_clients': int(stats_ventes['nb_clients'] or 0),
                    'total_depenses': depenses_total,
                    'benefice_net': benefice_net,
                    'marge_beneficiaire': round(marge, 2)
                },
                'top_produits': [dict(row) for row in top_produits],
                'repartition_paiements': [dict(row) for row in repartition_paiements]
            })
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la génération du résumé: {str(e)}")

# ============================================================================
# ROUTE 11: CRÉDITS EN RETARD
# ============================================================================

@app.get("/api/historique/credits-retard", tags=["Historique"])
async def get_credits_retard(jours_limite: int = 30):
    """
    Récupère les crédits qui dépassent un certain nombre de jours
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    id,
                    receipt_number,
                    sale_date,
                    customer_name,
                    customer_phone,
                    product_type,
                    total_amount,
                    CURRENT_DATE - sale_date as jours_ecoules
                FROM ventes
                WHERE payment_method = 'credit'
                AND (CURRENT_DATE - sale_date) > %s
                ORDER BY jours_ecoules DESC
            """, (jours_limite,))
            
            credits_retard = cursor.fetchall()
            cursor.close()
            
            return [convert_decimal_to_float(dict(row)) for row in credits_retard]
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des crédits en retard: {str(e)}")

# ============================================================================
# ROUTE 12: ANALYSE COMPARATIVE (Mois en cours vs Mois précédent)
# ============================================================================

@app.get("/api/historique/comparaison-mensuelle", tags=["Historique"])
async def get_comparaison_mensuelle():
    """
    Compare les performances du mois en cours avec le mois précédent
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Mois en cours
            cursor.execute("""
                SELECT 
                    COUNT(*) as nb_ventes,
                    SUM(total_amount) as ca_total,
                    SUM(quantity) as quantite_totale
                FROM ventes
                WHERE EXTRACT(YEAR FROM sale_date) = EXTRACT(YEAR FROM CURRENT_DATE)
                AND EXTRACT(MONTH FROM sale_date) = EXTRACT(MONTH FROM CURRENT_DATE)
            """)
            mois_actuel = cursor.fetchone()
            
            # Mois précédent
            cursor.execute("""
                SELECT 
                    COUNT(*) as nb_ventes,
                    SUM(total_amount) as ca_total,
                    SUM(quantity) as quantite_totale
                FROM ventes
                WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                AND sale_date < DATE_TRUNC('month', CURRENT_DATE)
            """)
            mois_precedent = cursor.fetchone()
            
            cursor.close()
            
            def calc_evolution(actuel, precedent):
                actuel = float(actuel or 0)
                precedent = float(precedent or 0)
                if precedent > 0:
                    return round(((actuel - precedent) / precedent) * 100, 1)
                return 0.0
            
            return convert_decimal_to_float({
                'mois_actuel': dict(mois_actuel),
                'mois_precedent': dict(mois_precedent),
                'evolution': {
                    'ventes': calc_evolution(mois_actuel['nb_ventes'], mois_precedent['nb_ventes']),
                    'ca': calc_evolution(mois_actuel['ca_total'], mois_precedent['ca_total']),
                    'quantite': calc_evolution(mois_actuel['quantite_totale'], mois_precedent['quantite_totale'])
                }
            })
            
        except Exception as e:
            cursor.close()
            raise HTTPException(status_code=500, detail=f"Erreur lors de la comparaison: {str(e)}")













# ============================================================================
# AJOUT À VOTRE FICHIER main.py - GESTION DES PAIEMENTS PARTIELS
# À ajouter après la création de la table ventes
# ============================================================================

# ============================================================================
# NOUVELLE TABLE POUR LES PAIEMENTS PARTIELS
# ============================================================================

def create_payments_table():
    """Créer la table des paiements partiels"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Table des paiements partiels
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS partial_payments (
                id SERIAL PRIMARY KEY,
                sale_id INTEGER NOT NULL REFERENCES ventes(id) ON DELETE CASCADE,
                receipt_number VARCHAR(50) NOT NULL,
                
                -- Montant du paiement
                payment_amount NUMERIC(10, 2) NOT NULL CHECK (payment_amount > 0),
                payment_method VARCHAR(20) NOT NULL CHECK (payment_method IN ('cash', 'mobile', 'bank')),
                
                -- Soldes avant/après
                balance_before NUMERIC(10, 2) NOT NULL CHECK (balance_before >= 0),
                balance_after NUMERIC(10, 2) NOT NULL CHECK (balance_after >= 0),
                
                -- Date et heure
                payment_date TIMESTAMP WITH TIME ZONE NOT NULL,
                payment_time VARCHAR(10) NOT NULL,
                
                -- Notes et caissier
                notes TEXT,
                cashier VARCHAR(255) NOT NULL,
                
                -- Metadata
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                
                CONSTRAINT check_payment_balance CHECK (balance_after = balance_before - payment_amount)
            );
            
            CREATE INDEX IF NOT EXISTS idx_partial_payments_sale ON partial_payments(sale_id);
            CREATE INDEX IF NOT EXISTS idx_partial_payments_date ON partial_payments(payment_date);
            CREATE INDEX IF NOT EXISTS idx_partial_payments_receipt ON partial_payments(receipt_number);
        """)
        
        cursor.close()

# Appeler cette fonction dans startup_event
# create_payments_table()

# ============================================================================
# MODÈLES PYDANTIC POUR LES PAIEMENTS PARTIELS
# ============================================================================

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class PartialPaymentCreate(BaseModel):
    sale_id: int = Field(..., gt=0)
    payment_amount: float = Field(..., gt=0)
    payment_method: str = Field(..., pattern=r'^(cash|mobile|bank)$')
    payment_date: datetime
    payment_time: str = Field(..., pattern=r'^\d{2}:\d{2}$')
    notes: Optional[str] = None
    cashier: str = Field(..., min_length=1, max_length=255)

    @field_validator('payment_amount')
    @classmethod
    def validate_payment_amount(cls, v, info):
        if v <= 0:
            raise ValueError('Le montant du paiement doit être positif')
        return v

class PartialPaymentResponse(BaseModel):
    id: int
    sale_id: int
    receipt_number: str
    payment_amount: float
    payment_method: str
    balance_before: float
    balance_after: float
    payment_date: datetime
    payment_time: str
    notes: Optional[str]
    cashier: str
    created_at: datetime

class SaleWithPayments(BaseModel):
    sale: dict
    total_paid: float
    remaining_balance: float
    payment_status: str
    payments: List[dict]

# ============================================================================
# SERVICES - PAIEMENTS PARTIELS
# ============================================================================

def get_sale_balance(sale_id: int) -> dict:
    """Obtenir le solde actuel d'une vente"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        # Récupérer la vente
        cursor.execute("""
            SELECT 
                id, receipt_number, customer_name, customer_phone,
                total_amount, payment_method, advance_amount
            FROM ventes
            WHERE id = %s
        """, (sale_id,))
        
        sale = cursor.fetchone()
        
        if not sale:
            cursor.close()
            raise ValueError("Vente non trouvée")
        
        # Calculer le montant initialement payé
        if sale['payment_method'] == 'advance':
            initial_paid = float(sale['advance_amount'] or 0)
        elif sale['payment_method'] == 'credit':
            initial_paid = 0
        else:
            initial_paid = float(sale['total_amount'])
        
        # Calculer le total des paiements partiels
        cursor.execute("""
            SELECT COALESCE(SUM(payment_amount), 0) as total_partial
            FROM partial_payments
            WHERE sale_id = %s
        """, (sale_id,))
        
        partial_result = cursor.fetchone()
        total_partial = float(partial_result['total_partial'])
        
        cursor.close()
        
        total_paid = initial_paid + total_partial
        remaining = float(sale['total_amount']) - total_paid
        
        return {
            'sale_id': sale_id,
            'receipt_number': sale['receipt_number'],
            'customer_name': sale['customer_name'],
            'total_amount': float(sale['total_amount']),
            'initial_paid': initial_paid,
            'total_partial_payments': total_partial,
            'total_paid': total_paid,
            'remaining_balance': max(0, remaining),
            'is_fully_paid': remaining <= 0
        }

def create_partial_payment(data: PartialPaymentCreate) -> dict:
    """Créer un paiement partiel"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        # Vérifier le solde actuel
        balance_info = get_sale_balance(data.sale_id)
        
        if balance_info['is_fully_paid']:
            cursor.close()
            raise ValueError("Cette vente est déjà entièrement payée")
        
        if data.payment_amount > balance_info['remaining_balance']:
            cursor.close()
            raise ValueError(f"Le montant du paiement ({data.payment_amount} F) dépasse le solde restant ({balance_info['remaining_balance']} F)")
        
        balance_before = balance_info['remaining_balance']
        balance_after = balance_before - data.payment_amount
        
        # Créer le paiement partiel
        cursor.execute("""
            INSERT INTO partial_payments (
                sale_id, receipt_number, payment_amount, payment_method,
                balance_before, balance_after, payment_date, payment_time,
                notes, cashier
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING *
        """, (
            data.sale_id, balance_info['receipt_number'], data.payment_amount,
            data.payment_method, balance_before, balance_after,
            data.payment_date, data.payment_time, data.notes, data.cashier
        ))
        
        result = cursor.fetchone()
        cursor.close()
        
        return convert_decimal_to_float(dict(result))

def get_sale_with_payments(sale_id: int) -> dict:
    """Obtenir une vente avec tous ses paiements"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        # Récupérer la vente
        cursor.execute("""
            SELECT 
                id, receipt_number, customer_name, customer_phone,
                product_type, quantity, unit_type, total_amount,
                payment_method, advance_amount, sale_date, sale_time
            FROM ventes
            WHERE id = %s
        """, (sale_id,))
        
        sale = cursor.fetchone()
        
        if not sale:
            cursor.close()
            raise ValueError("Vente non trouvée")
        
        # Récupérer tous les paiements partiels
        cursor.execute("""
            SELECT *
            FROM partial_payments
            WHERE sale_id = %s
            ORDER BY payment_date DESC, payment_time DESC
        """, (sale_id,))
        
        payments = cursor.fetchall()
        cursor.close()
        
        # Calculer les totaux
        balance_info = get_sale_balance(sale_id)
        
        # Déterminer le statut
        if balance_info['is_fully_paid']:
            status = 'completed'
        elif balance_info['total_paid'] > balance_info['initial_paid']:
            status = 'partial'
        else:
            status = 'pending'
        
        return convert_decimal_to_float({
            'sale': dict(sale),
            'initial_paid': balance_info['initial_paid'],
            'total_paid': balance_info['total_paid'],
            'remaining_balance': balance_info['remaining_balance'],
            'payment_status': status,
            'payments': [dict(p) for p in payments]
        })

def get_customer_summary(customer_name: str) -> dict:
    """Obtenir le résumé complet d'un client"""
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        # Récupérer toutes les ventes du client avec crédit ou avance
        cursor.execute("""
            SELECT id, receipt_number, total_amount, payment_method, advance_amount, sale_date
            FROM ventes
            WHERE customer_name = %s
            AND payment_method IN ('credit', 'advance')
            ORDER BY sale_date DESC
        """, (customer_name,))
        
        sales = cursor.fetchall()
        cursor.close()
        
        total_debt = 0
        sales_details = []
        
        for sale in sales:
            balance_info = get_sale_balance(sale['id'])
            
            if balance_info['remaining_balance'] > 0:
                total_debt += balance_info['remaining_balance']
                
                sales_details.append({
                    'receipt_number': sale['receipt_number'],
                    'sale_date': sale['sale_date'],
                    'total_amount': float(sale['total_amount']),
                    'remaining_balance': balance_info['remaining_balance'],
                    'payment_type': sale['payment_method']
                })
        
        return convert_decimal_to_float({
            'customer_name': customer_name,
            'total_outstanding_debt': total_debt,
            'number_of_unpaid_sales': len(sales_details),
            'unpaid_sales': sales_details
        })

# ============================================================================
# ROUTES API - PAIEMENTS PARTIELS
# ============================================================================

@app.post("/api/partial-payments", response_model=PartialPaymentResponse, status_code=201, tags=["Paiements"])
async def create_partial_payment_route(payment: PartialPaymentCreate):
    """
    Enregistrer un paiement partiel pour une vente à crédit ou avec avance
    """
    try:
        result = create_partial_payment(payment)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/sales/{sale_id}/balance", tags=["Paiements"])
async def get_sale_balance_route(sale_id: int):
    """
    Obtenir le solde actuel d'une vente
    """
    try:
        result = get_sale_balance(sale_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/sales/{sale_id}/payments", tags=["Paiements"])
async def get_sale_payments_route(sale_id: int):
    """
    Obtenir une vente avec tous ses paiements partiels
    """
    try:
        result = get_sale_with_payments(sale_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/customers/{customer_name}/summary", tags=["Paiements"])
async def get_customer_summary_route(customer_name: str):
    """
    Obtenir le résumé complet des dettes d'un client
    """
    try:
        result = get_customer_summary(customer_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/partial-payments/history", tags=["Paiements"])
async def get_payments_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    Obtenir l'historique de tous les paiements partiels
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                pp.*,
                v.customer_name,
                v.customer_phone,
                v.product_type,
                v.total_amount as sale_total
            FROM partial_payments pp
            JOIN ventes v ON pp.sale_id = v.id
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND pp.payment_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND pp.payment_date <= %s"
            params.append(end_date)
        
        query += " ORDER BY pp.payment_date DESC, pp.payment_time DESC OFFSET %s LIMIT %s"
        params.extend([skip, limit])
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        
        return [convert_decimal_to_float(dict(r)) for r in results]

# ============================================================================
# MISE À JOUR DE LA ROUTE HISTORIQUE POUR INCLURE LES SOLDES
# ============================================================================

@app.get("/api/historique/avances-detailed", tags=["Historique"])
async def get_ventes_avances_detailed():
    """
    Récupère toutes les ventes avec avances incluant les paiements partiels
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    id,
                    receipt_number,
                    sale_date,
                    sale_time,
                    customer_name,
                    customer_phone,
                    product_type,
                    quantity,
                    unit_type,
                    total_amount,
                    payment_method,
                    advance_amount
                FROM ventes
                WHERE payment_method = 'advance'
                ORDER BY sale_date DESC, sale_time DESC
            """)
            
            avances = cursor.fetchall()
            cursor.close()
            
            results = []
            for avance in avances:
                balance_info = get_sale_balance(avance['id'])
                
                result = convert_decimal_to_float(dict(avance))
                result['total_paid'] = balance_info['total_paid']
                result['remaining_balance'] = balance_info['remaining_balance']
                result['is_fully_paid'] = balance_info['is_fully_paid']
                result['payment_status'] = 'completed' if balance_info['is_fully_paid'] else 'partial'
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/historique/credits-detailed", tags=["Historique"])
async def get_ventes_credits_detailed():
    """
    Récupère toutes les ventes à crédit incluant les paiements partiels
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT 
                    id,
                    receipt_number,
                    sale_date,
                    sale_time,
                    customer_name,
                    customer_phone,
                    product_type,
                    quantity,
                    unit_type,
                    total_amount,
                    payment_method
                FROM ventes
                WHERE payment_method = 'credit'
                ORDER BY sale_date DESC, sale_time DESC
            """)
            
            credits = cursor.fetchall()
            cursor.close()
            
            results = []
            for credit in credits:
                balance_info = get_sale_balance(credit['id'])
                
                result = convert_decimal_to_float(dict(credit))
                result['total_paid'] = balance_info['total_paid']
                result['remaining_balance'] = balance_info['remaining_balance']
                result['is_fully_paid'] = balance_info['is_fully_paid']
                result['payment_status'] = 'completed' if balance_info['is_fully_paid'] else 'pending'
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/historique/customers-debts", tags=["Historique"])
async def get_customers_with_debts():
    """
    Récupère la liste de tous les clients avec leurs dettes totales
    """
    with get_db_connection() as db:
        cursor = db.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Récupérer tous les clients ayant des crédits ou avances
            cursor.execute("""
                SELECT DISTINCT customer_name, customer_phone
                FROM ventes
                WHERE payment_method IN ('credit', 'advance')
                AND customer_name IS NOT NULL
                AND customer_name != ''
                ORDER BY customer_name
            """)
            
            customers = cursor.fetchall()
            cursor.close()
            
            results = []
            for customer in customers:
                summary = get_customer_summary(customer['customer_name'])
                
                if summary['total_outstanding_debt'] > 0:
                    results.append({
                        'customer_name': customer['customer_name'],
                        'customer_phone': customer['customer_phone'],
                        'total_debt': summary['total_outstanding_debt'],
                        'number_of_debts': summary['number_of_unpaid_sales'],
                        'debts': summary['unpaid_sales']
                    })
            
            # Trier par montant de dette décroissant
            results.sort(key=lambda x: x['total_debt'], reverse=True)
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║         🌾 API GESTION CÉRÉALES 🌾                        ║
    ║                                                           ║
    ║  Configuration requise:                                   ║
    ║  1. PostgreSQL installé et en cours d'exécution          ║
    ║  2. Base de données 'cereales_db' créée                  ║
    ║  3. Modifier DATABASE_CONFIG au début du fichier         ║
    ║                                                           ║
    ║  Commandes PostgreSQL:                                    ║
    ║  CREATE DATABASE cereales_db;                            ║
    ║  CREATE USER postgres WITH PASSWORD 'votre_password';    ║
    ║  GRANT ALL PRIVILEGES ON DATABASE cereales_db TO postgres;║
    ║                                                           ║
    ║  Installation des dépendances:                           ║
    ║  pip install fastapi uvicorn psycopg2-binary             ║
    ║                                                           ║
    ║  Documentation: http://localhost:8000/docs               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )