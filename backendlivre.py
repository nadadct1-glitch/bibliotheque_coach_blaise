from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import datetime, date
import psycopg2
from psycopg2.extras import RealDictCursor
import base64
import uuid
import os
from contextlib import contextmanager
from pydantic import BaseModel, validator

# Configuration de la base de données uvicorn backendlivre:app --host 0.0.0.0 --port 8040 --reload
DB_CONFIG = {
    "host": "localhost",
    "database": "bibliotheque_db",
    "user": "postgres",
    "password": "beauty",
    "port": 5432
}

app = FastAPI(
    title="API Gestion de Bibliothèque",
    description="API pour gérer une bibliothèque personnelle",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ==================== MODÈLES PYDANTIC ====================

class BookBase(BaseModel):
    title: str
    author: str
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None
    pages: Optional[int] = None
    category: str
    language: Optional[str] = "francais"
    description: Optional[str] = None
    purchase_date: Optional[date] = None
    price: Optional[float] = None
    reading_time: Optional[float] = None
    rating: Optional[float] = None
    status: str = "to-read"
    copies_count: Optional[int] = 1
    
    @validator('rating')
    def validate_rating(cls, v):
        if v is not None and (v < 0 or v > 5):
            raise ValueError('La note doit être entre 0 et 5')
        return v
    
    @validator('publication_year')
    def validate_year(cls, v):
        if v is not None and (v < 1000 or v > datetime.now().year):
            raise ValueError(f'Année invalide')
        return v
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['to-read', 'reading', 'finished']
        if v not in valid_statuses:
            raise ValueError(f'Statut invalide. Doit être: {", ".join(valid_statuses)}')
        return v

    @validator('copies_count')
    def validate_copies_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Le nombre d\'exemplaires ne peut pas être négatif')
        return v
    
class BookCreate(BookBase):
    pass

class BookUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None
    pages: Optional[int] = None
    category: Optional[str] = None
    language: Optional[str] = None
    description: Optional[str] = None
    purchase_date: Optional[date] = None
    price: Optional[float] = None
    reading_time: Optional[float] = None
    rating: Optional[float] = None
    status: Optional[str] = None

class BookResponse(BookBase):
    id: int
    image_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class StatisticsResponse(BaseModel):
    total_books: int
    books_to_read: int
    books_reading: int
    books_finished: int
    total_pages: int
    average_rating: float
    total_spent: float
    books_by_category: dict
    books_by_language: dict

# ==================== GESTION BASE DE DONNÉES ====================

@contextmanager
def get_db_connection():
    """Context manager pour la connexion à la base de données"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialiser la base de données avec les tables nécessaires"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Créer la table des livres
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS books (
                id SERIAL PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                author VARCHAR(300) NOT NULL,
                isbn VARCHAR(50) UNIQUE,
                publisher VARCHAR(300),
                publication_year INTEGER,
                pages INTEGER CHECK (pages > 0),
                category VARCHAR(100) NOT NULL,
                language VARCHAR(50) DEFAULT 'francais',
                description TEXT,
                purchase_date DATE,
                price DECIMAL(10, 2) CHECK (price >= 0),
                reading_time DECIMAL(10, 2) CHECK (reading_time >= 0),
                rating DECIMAL(3, 1) CHECK (rating >= 0 AND rating <= 5),
                status VARCHAR(20) NOT NULL DEFAULT 'to-read' 
                    CHECK (status IN ('to-read', 'reading', 'finished')),
                image_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Ajouter la colonne copies_count
        cursor.execute("""
            ALTER TABLE books ADD COLUMN IF NOT EXISTS copies_count INTEGER DEFAULT 1 CHECK (copies_count >= 0);
            UPDATE books SET copies_count = 1 WHERE copies_count IS NULL;
        """)
        
        # Créer des index pour améliorer les performances
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);
            CREATE INDEX IF NOT EXISTS idx_books_author ON books(author);
            CREATE INDEX IF NOT EXISTS idx_books_category ON books(category);
            CREATE INDEX IF NOT EXISTS idx_books_status ON books(status);
            CREATE INDEX IF NOT EXISTS idx_books_created_at ON books(created_at DESC);
        """)
        
        # Créer un trigger pour mettre à jour updated_at automatiquement
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_books_updated_at ON books;
            CREATE TRIGGER update_books_updated_at
                BEFORE UPDATE ON books
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        # Créer la table des résumés
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id SERIAL PRIMARY KEY,
                book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                summary_type VARCHAR(50) NOT NULL DEFAULT 'general'
                    CHECK (summary_type IN ('general', 'chapter', 'detailed', 'analysis', 'personal')),
                content TEXT NOT NULL CHECK (LENGTH(content) >= 10),
                key_themes VARCHAR(500),
                main_takeaway TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(book_id, summary_type)
            );
        """)
        
        # Créer la table des citations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                id SERIAL PRIMARY KEY,
                book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                citation_text TEXT NOT NULL CHECK (LENGTH(citation_text) >= 3),
                page_number INTEGER CHECK (page_number > 0),
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Index pour les résumés et citations
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_summaries_book_id ON summaries(book_id);
            CREATE INDEX IF NOT EXISTS idx_summaries_type ON summaries(summary_type);
            CREATE INDEX IF NOT EXISTS idx_citations_book_id ON citations(book_id);
            CREATE INDEX IF NOT EXISTS idx_citations_page ON citations(page_number);
        """)
        
        # Triggers pour les résumés et citations
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_summaries_updated_at ON summaries;
            CREATE TRIGGER update_summaries_updated_at
                BEFORE UPDATE ON summaries
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_citations_updated_at ON citations;
            CREATE TRIGGER update_citations_updated_at
                BEFORE UPDATE ON citations
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        conn.commit()
        cursor.close()

# ==================== FONCTIONS UTILITAIRES ====================

def save_image(image_data: str) -> str:
    """Sauvegarder une image base64 et retourner l'URL"""
    try:
        # Extraire les données base64
        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data
        
        # Décoder l'image
        image_bytes = base64.b64decode(encoded)
        
        # Générer un nom de fichier unique
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join("uploads", filename)
        
        # Sauvegarder l'image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # return f"/uploads/{filename}"
        return f"http://localhost:8040/uploads/{filename}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la sauvegarde de l'image: {str(e)}")

def delete_image(image_url: str):
    """Supprimer une image du système de fichiers"""
    if image_url and image_url.startswith("/uploads/"):
        filepath = image_url.replace("/uploads/", "uploads/")
        if os.path.exists(filepath):
            os.remove(filepath)

# ==================== ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialiser la base de données au démarrage"""
    try:
        init_database()
        init_sorties_table()
        print("✅ Base de données initialisée avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de la base de données: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Servir la page HTML du formulaire"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/books", response_model=BookResponse, status_code=201)
async def create_book(
    title: str = Form(...),
    author: str = Form(...),
    category: str = Form(...),
    isbn: Optional[str] = Form(None),
    publisher: Optional[str] = Form(None),
    publication_year: Optional[int] = Form(None),
    pages: Optional[int] = Form(None),
    language: Optional[str] = Form("francais"),
    description: Optional[str] = Form(None),
    purchase_date: Optional[str] = Form(None),
    price: Optional[float] = Form(None),
    reading_time: Optional[float] = Form(None),
    rating: Optional[float] = Form(None),
    status: str = Form("to-read"),
    copies_count: Optional[int] = Form(1),
    image: Optional[str] = Form(None)
):
    """Créer un nouveau livre"""
    try:
        # Traiter l'image si fournie
        image_url = None
        if image:
            image_url = save_image(image)
        
        # Convertir purchase_date si fourni
        purchase_date_obj = None
        if purchase_date:
            try:
                purchase_date_obj = datetime.strptime(purchase_date, '%Y-%m-%d').date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Format de date invalide")
        
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                INSERT INTO books (
                    title, author, isbn, publisher, publication_year, pages,
                    category, language, description, purchase_date, price,
                    reading_time, rating, status, copies_count, image_url
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING *
            """, (
                title, author, isbn, publisher, publication_year, pages,
                category, language, description, purchase_date_obj, price,
                reading_time, rating, status, copies_count, image_url
            ))
            
            book = cursor.fetchone()
            cursor.close()
            
            return BookResponse(**book)
    
    except psycopg2.IntegrityError as e:
        if 'isbn' in str(e):
            raise HTTPException(status_code=400, detail="Un livre avec cet ISBN existe déjà")
        raise HTTPException(status_code=400, detail=f"Erreur d'intégrité: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/books", response_model=List[BookResponse])
async def get_books(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    order: str = "desc"
):
    """Récupérer la liste des livres avec filtres et pagination"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Construire la requête dynamiquement
            query = "SELECT * FROM books WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = %s"
                params.append(category)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            if search:
                query += " AND (title ILIKE %s OR author ILIKE %s OR description ILIKE %s)"
                search_param = f"%{search}%"
                params.extend([search_param, search_param, search_param])
            
            # Validation du tri
            valid_sort_fields = ['title', 'author', 'created_at', 'rating', 'publication_year']
            if sort_by not in valid_sort_fields:
                sort_by = 'created_at'
            
            valid_orders = ['asc', 'desc']
            if order.lower() not in valid_orders:
                order = 'desc'
            
            query += f" ORDER BY {sort_by} {order.upper()}"
            query += " LIMIT %s OFFSET %s"
            params.extend([limit, skip])
            
            cursor.execute(query, params)
            books = cursor.fetchall()
            cursor.close()
            
            return [BookResponse(**book) for book in books]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/books/{book_id}", response_model=BookResponse)
async def get_book(book_id: int):
    """Récupérer un livre par son ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM books WHERE id = %s", (book_id,))
            book = cursor.fetchone()
            cursor.close()
            
            if not book:
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            return BookResponse(**book)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")


@app.put("/api/books/{book_id}", response_model=BookResponse)
async def update_book(
    book_id: int,
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    isbn: Optional[str] = Form(None),
    publisher: Optional[str] = Form(None),
    publication_year: Optional[int] = Form(None),
    pages: Optional[int] = Form(None),
    category: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    purchase_date: Optional[str] = Form(None),
    price: Optional[float] = Form(None),
    reading_time: Optional[float] = Form(None),
    rating: Optional[float] = Form(None),
    status: Optional[str] = Form(None),
    copies_count: Optional[int] = Form(None),  # ← AJOUTER CETTE LIGNE
    image: Optional[str] = Form(None)
):
    """Mettre à jour un livre"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le livre existe
            cursor.execute("SELECT * FROM books WHERE id = %s", (book_id,))
            existing_book = cursor.fetchone()
            
            if not existing_book:
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            # Construire la requête de mise à jour dynamiquement
            update_fields = []
            params = []
            
            if title is not None:
                update_fields.append("title = %s")
                params.append(title)
            if author is not None:
                update_fields.append("author = %s")
                params.append(author)
            if isbn is not None:
                update_fields.append("isbn = %s")
                params.append(isbn)
            if publisher is not None:
                update_fields.append("publisher = %s")
                params.append(publisher)
            if publication_year is not None:
                update_fields.append("publication_year = %s")
                params.append(publication_year)
            if pages is not None:
                update_fields.append("pages = %s")
                params.append(pages)
            if category is not None:
                update_fields.append("category = %s")
                params.append(category)
            if language is not None:
                update_fields.append("language = %s")
                params.append(language)
            if description is not None:
                update_fields.append("description = %s")
                params.append(description)
            if purchase_date is not None:
                purchase_date_obj = datetime.strptime(purchase_date, '%Y-%m-%d').date()
                update_fields.append("purchase_date = %s")
                params.append(purchase_date_obj)
            if price is not None:
                update_fields.append("price = %s")
                params.append(price)
            if reading_time is not None:
                update_fields.append("reading_time = %s")
                params.append(reading_time)
            if rating is not None:
                update_fields.append("rating = %s")
                params.append(rating)
            if status is not None:
                update_fields.append("status = %s")
                params.append(status)
            if copies_count is not None:
                update_fields.append("copies_count = %s")
                params.append(copies_count)
            # Gérer l'image
            if image is not None:
                # Supprimer l'ancienne image si elle existe
                if existing_book['image_url']:
                    delete_image(existing_book['image_url'])
                # Sauvegarder la nouvelle image
                image_url = save_image(image)
                update_fields.append("image_url = %s")
                params.append(image_url)
            
            if not update_fields:
                raise HTTPException(status_code=400, detail="Aucun champ à mettre à jour")
            
            params.append(book_id)
            query = f"UPDATE books SET {', '.join(update_fields)} WHERE id = %s RETURNING *"
            
            cursor.execute(query, params)
            updated_book = cursor.fetchone()
            cursor.close()
            
            return BookResponse(**updated_book)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.delete("/api/books/{book_id}")
async def delete_book(book_id: int):
    """Supprimer un livre"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Récupérer le livre pour supprimer l'image
            cursor.execute("SELECT image_url FROM books WHERE id = %s", (book_id,))
            book = cursor.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            # Supprimer l'image si elle existe
            if book['image_url']:
                delete_image(book['image_url'])
            
            # Supprimer le livre
            cursor.execute("DELETE FROM books WHERE id = %s", (book_id,))
            cursor.close()
            
            return {"message": "Livre supprimé avec succès"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """Récupérer les statistiques de la bibliothèque"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Statistiques générales
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_books,
                    COUNT(*) FILTER (WHERE status = 'to-read') as books_to_read,
                    COUNT(*) FILTER (WHERE status = 'reading') as books_reading,
                    COUNT(*) FILTER (WHERE status = 'finished') as books_finished,
                    COALESCE(SUM(pages), 0) as total_pages,
                    COALESCE(AVG(rating), 0) as average_rating,
                    COALESCE(SUM(price), 0) as total_spent
                FROM books
            """)
            stats = cursor.fetchone()
            
            # Livres par catégorie
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM books
                GROUP BY category
                ORDER BY count DESC
            """)
            books_by_category = {row['category']: row['count'] for row in cursor.fetchall()}
            
            # Livres par langue
            cursor.execute("""
                SELECT language, COUNT(*) as count
                FROM books
                GROUP BY language
                ORDER BY count DESC
            """)
            books_by_language = {row['language']: row['count'] for row in cursor.fetchall()}
            
            cursor.close()
            
            return StatisticsResponse(
                total_books=stats['total_books'],
                books_to_read=stats['books_to_read'],
                books_reading=stats['books_reading'],
                books_finished=stats['books_finished'],
                total_pages=stats['total_pages'],
                average_rating=float(stats['average_rating']),
                total_spent=float(stats['total_spent']),
                books_by_category=books_by_category,
                books_by_language=books_by_language
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/categories")
async def get_categories():
    """Récupérer toutes les catégories disponibles"""
    categories = [
        "devperso", "education", "sante", "bonheur", "emploi", "leadership",
        "entreprenariat", "management", "prosperite", "literature", "academie",
        "romance", "thriller", "fantasy", "science-fiction", "historique",
        "biographie", "essai", "poesie", "jeunesse", "autre"
    ]
    return {"categories": categories}

@app.get("/api/health")
async def health_check():
    """Vérifier l'état de l'API et de la base de données"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }








# ==================== CODE À AJOUTER DANS main.py ====================
# Copiez ce code APRÈS les routes existantes des livres, AVANT le if __name__ == "__main__"

# ==================== MODÈLES PYDANTIC POUR RÉSUMÉS ET CITATIONS ====================

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, validator

# Modèle pour les résumés
class SummaryBase(BaseModel):
    book_id: int
    summary_type: str = "general"
    content: str
    key_themes: Optional[str] = None
    main_takeaway: Optional[str] = None
    
    @validator('summary_type')
    def validate_summary_type(cls, v):
        valid_types = ['general', 'chapter', 'detailed', 'analysis', 'personal']
        if v not in valid_types:
            raise ValueError(f'Type de résumé invalide. Doit être: {", ".join(valid_types)}')
        return v
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Le résumé doit contenir au moins 10 caractères')
        if len(v) > 50000:
            raise ValueError('Le résumé ne peut pas dépasser 50000 caractères')
        return v.strip()

class SummaryCreate(SummaryBase):
    pass

class SummaryUpdate(BaseModel):
    summary_type: Optional[str] = None
    content: Optional[str] = None
    key_themes: Optional[str] = None
    main_takeaway: Optional[str] = None

class SummaryResponse(SummaryBase):
    id: int
    created_at: datetime
    updated_at: datetime

# Modèle pour les citations
class CitationBase(BaseModel):
    book_id: int
    citation_text: str
    page_number: Optional[int] = None
    context: Optional[str] = None
    
    @validator('citation_text')
    def validate_citation_text(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('La citation doit contenir au moins 3 caractères')
        if len(v) > 2000:
            raise ValueError('La citation ne peut pas dépasser 2000 caractères')
        return v.strip()
    
    @validator('page_number')
    def validate_page_number(cls, v):
        if v is not None and v < 1:
            raise ValueError('Le numéro de page doit être positif')
        return v

class CitationCreate(CitationBase):
    pass

class CitationUpdate(BaseModel):
    citation_text: Optional[str] = None
    page_number: Optional[int] = None
    context: Optional[str] = None

class CitationResponse(CitationBase):
    id: int
    created_at: datetime
    updated_at: datetime

# ==================== INITIALISATION DES TABLES ====================

# IMPORTANT: Ajoutez ces créations de tables dans la fonction init_database()
# Cherchez la fonction init_database() dans votre main.py et ajoutez ceci APRÈS la création de la table books:

"""
# Créer la table des résumés
cursor.execute('''
    CREATE TABLE IF NOT EXISTS summaries (
        id SERIAL PRIMARY KEY,
        book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
        summary_type VARCHAR(50) NOT NULL DEFAULT 'general'
            CHECK (summary_type IN ('general', 'chapter', 'detailed', 'analysis', 'personal')),
        content TEXT NOT NULL CHECK (LENGTH(content) >= 10),
        key_themes VARCHAR(500),
        main_takeaway TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(book_id, summary_type)
    );
''')

# Créer la table des citations
cursor.execute('''
    CREATE TABLE IF NOT EXISTS citations (
        id SERIAL PRIMARY KEY,
        book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
        citation_text TEXT NOT NULL CHECK (LENGTH(citation_text) >= 3),
        page_number INTEGER CHECK (page_number > 0),
        context TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
''')

# Créer des index pour améliorer les performances
cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_summaries_book_id ON summaries(book_id);
    CREATE INDEX IF NOT EXISTS idx_summaries_type ON summaries(summary_type);
    CREATE INDEX IF NOT EXISTS idx_citations_book_id ON citations(book_id);
    CREATE INDEX IF NOT EXISTS idx_citations_page ON citations(page_number);
''')

# Créer les triggers pour updated_at
cursor.execute('''
    DROP TRIGGER IF EXISTS update_summaries_updated_at ON summaries;
    CREATE TRIGGER update_summaries_updated_at
        BEFORE UPDATE ON summaries
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
''')

cursor.execute('''
    DROP TRIGGER IF EXISTS update_citations_updated_at ON citations;
    CREATE TRIGGER update_citations_updated_at
        BEFORE UPDATE ON citations
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
''')
"""

# ==================== ENDPOINTS POUR LES RÉSUMÉS ====================

@app.post("/api/summaries", response_model=SummaryResponse, status_code=201)
async def create_summary(summary: SummaryCreate):
    """Créer un nouveau résumé de livre"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le livre existe
            cursor.execute("SELECT id FROM books WHERE id = %s", (summary.book_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            # Vérifier si un résumé de ce type existe déjà pour ce livre
            cursor.execute(
                "SELECT id FROM summaries WHERE book_id = %s AND summary_type = %s",
                (summary.book_id, summary.summary_type)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Mettre à jour le résumé existant
                cursor.execute("""
                    UPDATE summaries 
                    SET content = %s, key_themes = %s, main_takeaway = %s
                    WHERE id = %s
                    RETURNING *
                """, (summary.content, summary.key_themes, summary.main_takeaway, existing['id']))
            else:
                # Créer un nouveau résumé
                cursor.execute("""
                    INSERT INTO summaries (book_id, summary_type, content, key_themes, main_takeaway)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING *
                """, (summary.book_id, summary.summary_type, summary.content, 
                      summary.key_themes, summary.main_takeaway))
            
            result = cursor.fetchone()
            cursor.close()
            
            return SummaryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/summaries", response_model=List[SummaryResponse])
async def get_summaries(
    skip: int = 0,
    limit: int = 100,
    summary_type: Optional[str] = None
):
    """Récupérer la liste des résumés"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM summaries WHERE 1=1"
            params = []
            
            if summary_type:
                query += " AND summary_type = %s"
                params.append(summary_type)
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, skip])
            
            cursor.execute(query, params)
            summaries = cursor.fetchall()
            cursor.close()
            
            return [SummaryResponse(**summary) for summary in summaries]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/summaries/{summary_id}", response_model=SummaryResponse)
async def get_summary(summary_id: int):
    """Récupérer un résumé par son ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM summaries WHERE id = %s", (summary_id,))
            summary = cursor.fetchone()
            cursor.close()
            
            if not summary:
                raise HTTPException(status_code=404, detail="Résumé non trouvé")
            
            return SummaryResponse(**summary)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/summaries/book/{book_id}", response_model=List[SummaryResponse])
async def get_summaries_by_book(book_id: int):
    """Récupérer tous les résumés d'un livre"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le livre existe
            cursor.execute("SELECT id FROM books WHERE id = %s", (book_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            cursor.execute(
                "SELECT * FROM summaries WHERE book_id = %s ORDER BY created_at DESC",
                (book_id,)
            )
            summaries = cursor.fetchall()
            cursor.close()
            
            return [SummaryResponse(**summary) for summary in summaries]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.put("/api/summaries/{summary_id}", response_model=SummaryResponse)
async def update_summary(summary_id: int, summary: SummaryUpdate):
    """Mettre à jour un résumé"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le résumé existe
            cursor.execute("SELECT * FROM summaries WHERE id = %s", (summary_id,))
            existing = cursor.fetchone()
            
            if not existing:
                raise HTTPException(status_code=404, detail="Résumé non trouvé")
            
            # Construire la requête de mise à jour
            update_fields = []
            params = []
            
            if summary.summary_type is not None:
                update_fields.append("summary_type = %s")
                params.append(summary.summary_type)
            if summary.content is not None:
                update_fields.append("content = %s")
                params.append(summary.content)
            if summary.key_themes is not None:
                update_fields.append("key_themes = %s")
                params.append(summary.key_themes)
            if summary.main_takeaway is not None:
                update_fields.append("main_takeaway = %s")
                params.append(summary.main_takeaway)
            
            if not update_fields:
                raise HTTPException(status_code=400, detail="Aucun champ à mettre à jour")
            
            params.append(summary_id)
            query = f"UPDATE summaries SET {', '.join(update_fields)} WHERE id = %s RETURNING *"
            
            cursor.execute(query, params)
            updated_summary = cursor.fetchone()
            cursor.close()
            
            return SummaryResponse(**updated_summary)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.delete("/api/summaries/{summary_id}")
async def delete_summary(summary_id: int):
    """Supprimer un résumé"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le résumé existe
            cursor.execute("SELECT id FROM summaries WHERE id = %s", (summary_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Résumé non trouvé")
            
            cursor.execute("DELETE FROM summaries WHERE id = %s", (summary_id,))
            cursor.close()
            
            return {"message": "Résumé supprimé avec succès"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

# ==================== ENDPOINTS POUR LES CITATIONS ====================

@app.post("/api/citations", response_model=CitationResponse, status_code=201)
async def create_citation(citation: CitationCreate):
    """Créer une nouvelle citation"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le livre existe
            cursor.execute("SELECT id FROM books WHERE id = %s", (citation.book_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            cursor.execute("""
                INSERT INTO citations (book_id, citation_text, page_number, context)
                VALUES (%s, %s, %s, %s)
                RETURNING *
            """, (citation.book_id, citation.citation_text, citation.page_number, citation.context))
            
            result = cursor.fetchone()
            cursor.close()
            
            return CitationResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/citations", response_model=List[CitationResponse])
async def get_citations(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None
):
    """Récupérer la liste des citations"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT * FROM citations WHERE 1=1"
            params = []
            
            if search:
                query += " AND citation_text ILIKE %s"
                params.append(f"%{search}%")
            
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, skip])
            
            cursor.execute(query, params)
            citations = cursor.fetchall()
            cursor.close()
            
            return [CitationResponse(**citation) for citation in citations]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/citations/{citation_id}", response_model=CitationResponse)
async def get_citation(citation_id: int):
    """Récupérer une citation par son ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM citations WHERE id = %s", (citation_id,))
            citation = cursor.fetchone()
            cursor.close()
            
            if not citation:
                raise HTTPException(status_code=404, detail="Citation non trouvée")
            
            return CitationResponse(**citation)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/api/citations/book/{book_id}", response_model=List[CitationResponse])
async def get_citations_by_book(book_id: int):
    """Récupérer toutes les citations d'un livre"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que le livre existe
            cursor.execute("SELECT id FROM books WHERE id = %s", (book_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            cursor.execute(
                "SELECT * FROM citations WHERE book_id = %s ORDER BY page_number NULLS LAST, created_at ASC",
                (book_id,)
            )
            citations = cursor.fetchall()
            cursor.close()
            
            return [CitationResponse(**citation) for citation in citations]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.put("/api/citations/{citation_id}", response_model=CitationResponse)
async def update_citation(citation_id: int, citation: CitationUpdate):
    """Mettre à jour une citation"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que la citation existe
            cursor.execute("SELECT * FROM citations WHERE id = %s", (citation_id,))
            existing = cursor.fetchone()
            
            if not existing:
                raise HTTPException(status_code=404, detail="Citation non trouvée")
            
            # Construire la requête de mise à jour
            update_fields = []
            params = []
            
            if citation.citation_text is not None:
                update_fields.append("citation_text = %s")
                params.append(citation.citation_text)
            if citation.page_number is not None:
                update_fields.append("page_number = %s")
                params.append(citation.page_number)
            if citation.context is not None:
                update_fields.append("context = %s")
                params.append(citation.context)
            
            if not update_fields:
                raise HTTPException(status_code=400, detail="Aucun champ à mettre à jour")
            
            params.append(citation_id)
            query = f"UPDATE citations SET {', '.join(update_fields)} WHERE id = %s RETURNING *"
            
            cursor.execute(query, params)
            updated_citation = cursor.fetchone()
            cursor.close()
            
            return CitationResponse(**updated_citation)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.delete("/api/citations/{citation_id}")
async def delete_citation(citation_id: int):
    """Supprimer une citation"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier que la citation existe
            cursor.execute("SELECT id FROM citations WHERE id = %s", (citation_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="Citation non trouvée")
            
            cursor.execute("DELETE FROM citations WHERE id = %s", (citation_id,))
            cursor.close()
            
            return {"message": "Citation supprimée avec succès"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

# ==================== ENDPOINT POUR SERVIR LA PAGE HTML ====================

@app.get("/add_summary_citation.html", response_class=HTMLResponse)
async def read_summary_citation_page():
    """Servir la page HTML du formulaire résumés et citations"""
    with open("templates/add_summary_citation.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ==================== FIN DU CODE À AJOUTER ====================






# ==================== CODE À AJOUTER DANS main.py ====================
# Copiez ce code APRÈS les routes existantes

# ==================== ROUTE POUR SERVIR LA PAGE BIBLIOTHÈQUE ====================

@app.get("/bibliotheque.html", response_class=HTMLResponse)
async def read_bibliotheque_page():
    """Servir la page HTML de la bibliothèque"""
    with open("templates/bibliotheque.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ==================== FIN DU CODE À AJOUTER ====================










# ==================== CODE API SORTIES DE LIVRES - VERSION CORRIGÉE ====================
# À AJOUTER DANS backendlivre.py APRÈS les routes existantes

from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel, validator

# ==================== MODÈLES PYDANTIC ====================

class SortieLivreBase(BaseModel):
    book_id: int
    transaction_type: str
    client_name: str
    client_phone: str
    client_email: Optional[str] = None
    client_address: Optional[str] = None
    quantity: int = 1
    sale_price: Optional[float] = None
    payment_method: Optional[str] = None
    transaction_date: date
    return_date: Optional[date] = None
    notes: Optional[str] = None
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        valid_types = ['vente', 'pret', 'don']
        if v not in valid_types:
            raise ValueError(f'Type invalide. Doit être: {", ".join(valid_types)}')
        return v

class SortieLivreCreate(SortieLivreBase):
    pass

class SortieLivreResponse(SortieLivreBase):
    id: int
    returned: bool
    created_at: datetime
    updated_at: datetime
    book_title: Optional[str] = None
    book_author: Optional[str] = None

# ==================== FONCTION D'INITIALISATION DE LA TABLE ====================

def init_sorties_table():
    """Créer la table sorties_livres au démarrage"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sorties_livres (
                    id SERIAL PRIMARY KEY,
                    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE RESTRICT,
                    transaction_type VARCHAR(20) NOT NULL CHECK (transaction_type IN ('vente', 'pret', 'don')),
                    client_name VARCHAR(200) NOT NULL,
                    client_phone VARCHAR(20) NOT NULL,
                    client_email VARCHAR(100),
                    client_address VARCHAR(500),
                    quantity INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
                    sale_price DECIMAL(10, 2),
                    payment_method VARCHAR(20),
                    transaction_date DATE NOT NULL,
                    return_date DATE,
                    notes TEXT,
                    returned BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_sorties_book_id ON sorties_livres(book_id);
                CREATE INDEX IF NOT EXISTS idx_sorties_transaction_type ON sorties_livres(transaction_type);
                CREATE INDEX IF NOT EXISTS idx_sorties_returned ON sorties_livres(returned);
                
                DROP TRIGGER IF EXISTS update_sorties_livres_updated_at ON sorties_livres;
                CREATE TRIGGER update_sorties_livres_updated_at
                    BEFORE UPDATE ON sorties_livres
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """)
            
            conn.commit()
            cursor.close()
            print("✅ Table sorties_livres créée avec succès")
    except Exception as e:
        print(f"❌ Erreur création table sorties: {e}")

# IMPORTANT: Appelez init_sorties_table() dans @app.on_event("startup")
# Ajoutez cette ligne après init_database() dans la fonction startup_event:
#     init_sorties_table()

# ==================== ROUTES API ====================

@app.post("/api/sorties", response_model=SortieLivreResponse, status_code=201)
async def create_sortie_livre(sortie: SortieLivreCreate):
    """Créer une sortie et réduire le stock automatiquement"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Vérifier le livre et le stock
            cursor.execute("SELECT id, title, author, copies_count FROM books WHERE id = %s", (sortie.book_id,))
            book = cursor.fetchone()
            
            if not book:
                raise HTTPException(status_code=404, detail="Livre non trouvé")
            
            current_stock = book['copies_count'] or 0
            if current_stock < sortie.quantity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Stock insuffisant. Disponible: {current_stock}, Demandé: {sortie.quantity}"
                )
            
            # Insérer la sortie
            cursor.execute("""
                INSERT INTO sorties_livres (
                    book_id, transaction_type, client_name, client_phone,
                    client_email, client_address, quantity, sale_price,
                    payment_method, transaction_date, return_date, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (
                sortie.book_id, sortie.transaction_type, sortie.client_name,
                sortie.client_phone, sortie.client_email, sortie.client_address,
                sortie.quantity, sortie.sale_price, sortie.payment_method,
                sortie.transaction_date, sortie.return_date, sortie.notes
            ))
            
            result = cursor.fetchone()
            
            # Réduire le stock
            new_stock = current_stock - sortie.quantity
            cursor.execute(
                "UPDATE books SET copies_count = %s WHERE id = %s",
                (new_stock, sortie.book_id)
            )
            
            conn.commit()
            cursor.close()
            
            # Préparer la réponse
            response_data = dict(result)
            response_data['book_title'] = book['title']
            response_data['book_author'] = book['author']
            
            return SortieLivreResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/sorties/statistics")
async def get_sortie_statistics():
    """Récupérer les statistiques EN TEMPS RÉEL"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Statistiques avec COALESCE pour éviter les NULL
            cursor.execute("""
                SELECT 
                    COALESCE(COUNT(*), 0) as total_sorties,
                    COALESCE(COUNT(*) FILTER (WHERE transaction_type = 'vente'), 0) as total_ventes,
                    COALESCE(COUNT(*) FILTER (WHERE transaction_type = 'pret'), 0) as total_prets,
                    COALESCE(COUNT(*) FILTER (WHERE transaction_type = 'don'), 0) as total_dons,
                    COALESCE(COUNT(*) FILTER (WHERE transaction_type = 'pret' AND returned = FALSE), 0) as prets_en_cours,
                    COALESCE(COUNT(*) FILTER (WHERE transaction_type = 'pret' AND returned = TRUE), 0) as prets_retournes,
                    COALESCE(SUM(COALESCE(sale_price, 0) * COALESCE(quantity, 1)) FILTER (WHERE transaction_type = 'vente'), 0) as revenue_total
                FROM sorties_livres
            """)
            stats = cursor.fetchone()
            cursor.close()
            
            # Retourner un dictionnaire simple (PAS de Pydantic model)
            return {
                "total_sorties": int(stats['total_sorties']),
                "total_ventes": int(stats['total_ventes']),
                "total_prets": int(stats['total_prets']),
                "total_dons": int(stats['total_dons']),
                "prets_en_cours": int(stats['prets_en_cours']),
                "prets_retournes": int(stats['prets_retournes']),
                "revenue_total": float(stats['revenue_total']),
                "revenue_par_mois": {},
                "sorties_par_livre": {}
            }
    
    except Exception as e:
        print(f"ERREUR STATS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/api/sorties", response_model=List[SortieLivreResponse])
async def get_sorties(skip: int = 0, limit: int = 100):
    """Liste des sorties"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT s.*, b.title as book_title, b.author as book_author
                FROM sorties_livres s
                LEFT JOIN books b ON s.book_id = b.id
                ORDER BY s.transaction_date DESC
                LIMIT %s OFFSET %s
            """, (limit, skip))
            
            sorties = cursor.fetchall()
            cursor.close()
            
            return [SortieLivreResponse(**sortie) for sortie in sorties]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/sortie_livre.html", response_class=HTMLResponse)
async def read_sortie_livre_page():
    """Servir la page HTML"""
    with open("templates/sortie_livre.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ==================== FIN DU CODE ================







# ==================== CODE API JEU DE CITATIONS ====================
# À COPIER-COLLER DANS backendlivre.py APRÈS les routes des sorties

import random

# ==================== ENDPOINT POUR LE JEU DE CITATIONS ====================

@app.get("/api/jeu/citation-aleatoire")
async def get_citation_aleatoire(exclude_ids: Optional[str] = None):
    """
    Récupérer une citation aléatoire avec les infos du livre
    exclude_ids: IDs de citations déjà vues (séparés par des virgules)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Construire la requête pour exclure les citations déjà vues
            query = """
                SELECT 
                    c.id as citation_id,
                    c.citation_text,
                    c.page_number,
                    b.id as book_id,
                    b.title,
                    b.author,
                    b.image_url,
                    b.description,
                    b.category,
                    b.price,
                    b.copies_count,
                    b.rating,
                    b.pages
                FROM citations c
                INNER JOIN books b ON c.book_id = b.id
                WHERE 1=1
            """
            params = []
            
            # Exclure les citations déjà vues - CORRECTION: Gestion sécurisée
            if exclude_ids and exclude_ids.strip():
                try:
                    # Filtrer et convertir les IDs valides
                    excluded = []
                    for id_str in exclude_ids.split(','):
                        id_str = id_str.strip()
                        if id_str and id_str.isdigit():
                            excluded.append(int(id_str))
                    
                    if excluded:
                        placeholders = ','.join(['%s'] * len(excluded))
                        query += f" AND c.id NOT IN ({placeholders})"
                        params.extend(excluded)
                except Exception as parse_error:
                    print(f"Erreur parsing exclude_ids: {parse_error}")
                    # Continuer sans exclusion si erreur de parsing
            
            cursor.execute(query, params)
            citations = cursor.fetchall()
            cursor.close()
            
            if not citations or len(citations) == 0:
                return {
                    "success": False,
                    "message": "Aucune citation disponible",
                    "citation": None
                }
            
            # Sélectionner une citation aléatoire
            citation = random.choice(citations)
            
            return {
                "success": True,
                "citation": {
                    "citation_id": citation['citation_id'],
                    "citation_text": citation['citation_text'],
                    "page_number": citation['page_number'],
                    "book": {
                        "id": citation['book_id'],
                        "title": citation['title'],
                        "author": citation['author'],
                        "image_url": citation['image_url'],
                        "description": citation['description'],
                        "category": citation['category'],
                        "price": float(citation['price']) if citation['price'] else None,
                        "copies_count": citation['copies_count'],
                        "rating": float(citation['rating']) if citation['rating'] else None,
                        "pages": citation['pages']
                    }
                }
            }
    
    except Exception as e:
        print(f"ERREUR API CITATION: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/jeu_citations.html", response_class=HTMLResponse)
async def read_jeu_citations_page():
    """Servir la page HTML du jeu de citations"""
    with open("templates/jeu_citations.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ==================== FIN CODE API JEU ====================






@app.get("/adventure_reading.html", response_class=HTMLResponse)
async def read_adventure():
    with open("templates/adventure_reading.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())



















# ==================== API DASHBOARD BIBLIOTHÈQUE ====================
# À COPIER-COLLER DANS backendlivre.py

from datetime import datetime, timedelta
from typing import Optional

# ==================== ROUTES DASHBOARD ====================

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """
    Récupérer toutes les statistiques pour le dashboard
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. STATISTIQUES LIVRES
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_books,
                    SUM(copies_count) as total_copies,
                    COALESCE(SUM(price), 0) as total_value,
                    COALESCE(AVG(rating), 0) as avg_rating,
                    COALESCE(SUM(pages), 0) as total_pages
                FROM books
            """)
            books_stats = cursor.fetchone()
            
            # 2. STATISTIQUES CITATIONS
            cursor.execute("SELECT COUNT(*) as total_citations FROM citations")
            citations_stats = cursor.fetchone()
            
            # 3. STATISTIQUES RÉSUMÉS
            cursor.execute("SELECT COUNT(*) as total_resumes FROM summaries")
            resumes_stats = cursor.fetchone()
            
            # 4. STATISTIQUES SORTIES
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_sorties,
                    COUNT(*) FILTER (WHERE transaction_type = 'vente') as total_ventes,
                    COUNT(*) FILTER (WHERE transaction_type = 'pret') as total_prets,
                    COUNT(*) FILTER (WHERE transaction_type = 'don') as total_dons,
                    COUNT(*) FILTER (WHERE transaction_type = 'pret' AND returned = FALSE) as prets_en_cours,
                    COALESCE(SUM(sale_price * quantity) FILTER (WHERE transaction_type = 'vente'), 0) as revenue_total,
                    COALESCE(SUM(quantity), 0) as total_quantity
                FROM sorties_livres
            """)
            sorties_stats = cursor.fetchone()
            
            # 5. LIVRES AJOUTÉS CE MOIS
            cursor.execute("""
                SELECT COUNT(*) as books_this_month
                FROM books
                WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)
            """)
            monthly_books = cursor.fetchone()
            
            # 6. REVENUS CE MOIS
            cursor.execute("""
                SELECT COALESCE(SUM(sale_price * quantity), 0) as revenue_this_month
                FROM sorties_livres
                WHERE transaction_type = 'vente'
                AND transaction_date >= DATE_TRUNC('month', CURRENT_DATE)
            """)
            monthly_revenue = cursor.fetchone()
            
            # 7. CATÉGORIES LES PLUS POPULAIRES
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM books
                WHERE category IS NOT NULL
                GROUP BY category
                ORDER BY count DESC
                LIMIT 10
            """)
            top_categories = cursor.fetchall()
            
            # 8. ESTIMATION DU TEMPS DE LECTURE (moyenne 250 mots/min, 300 mots/page)
            total_pages = books_stats['total_pages'] or 0
            reading_time_minutes = (total_pages * 300) / 250
            reading_time_hours = round(reading_time_minutes / 60, 1)
            
            cursor.close()
            
            return {
                "books": {
                    "total": int(books_stats['total_books']),
                    "total_copies": int(books_stats['total_copies'] or 0),
                    "total_value": float(books_stats['total_value'] or 0),
                    "avg_rating": round(float(books_stats['avg_rating'] or 0), 1),
                    "total_pages": int(books_stats['total_pages'] or 0),
                    "this_month": int(monthly_books['books_this_month'])
                },
                "citations": {
                    "total": int(citations_stats['total_citations'])
                },
                "resumes": {
                    "total": int(resumes_stats['total_resumes'])
                },
                "sorties": {
                    "total": int(sorties_stats['total_sorties'] or 0),
                    "ventes": int(sorties_stats['total_ventes'] or 0),
                    "prets": int(sorties_stats['total_prets'] or 0),
                    "dons": int(sorties_stats['total_dons'] or 0),
                    "prets_en_cours": int(sorties_stats['prets_en_cours'] or 0),
                    "total_quantity": int(sorties_stats['total_quantity'] or 0)
                },
                "finance": {
                    "revenue_total": float(sorties_stats['revenue_total'] or 0),
                    "revenue_this_month": float(monthly_revenue['revenue_this_month'] or 0),
                    "total_invested": float(books_stats['total_value'] or 0),
                    "avg_price_per_book": round(float(books_stats['total_value'] or 0) / max(int(books_stats['total_books']), 1), 2)
                },
                "reading": {
                    "estimated_hours": reading_time_hours,
                    "estimated_days": round(reading_time_hours / 24, 1)
                },
                "categories": [
                    {"name": cat['category'], "count": int(cat['count'])}
                    for cat in top_categories
                ]
            }
    
    except Exception as e:
        print(f"ERREUR DASHBOARD STATS: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/books-evolution")
async def get_books_evolution(period: str = "year"):
    """
    Évolution des ajouts de livres par mois
    period: 'year' ou 'month'
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if period == "year":
                # Derniers 12 mois
                cursor.execute("""
                    SELECT 
                        TO_CHAR(created_at, 'Mon') as month,
                        EXTRACT(MONTH FROM created_at) as month_num,
                        COUNT(*) as count
                    FROM books
                    WHERE created_at >= CURRENT_DATE - INTERVAL '12 months'
                    GROUP BY TO_CHAR(created_at, 'Mon'), EXTRACT(MONTH FROM created_at)
                    ORDER BY month_num
                """)
            else:
                # Dernier mois par jour
                cursor.execute("""
                    SELECT 
                        TO_CHAR(created_at, 'DD/MM') as day,
                        COUNT(*) as count
                    FROM books
                    WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY TO_CHAR(created_at, 'DD/MM'), created_at
                    ORDER BY created_at
                """)
            
            results = cursor.fetchall()
            cursor.close()
            
            return {
                "labels": [r['month'] if period == "year" else r['day'] for r in results],
                "data": [int(r['count']) for r in results]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/sorties-evolution")
async def get_sorties_evolution(period: str = "year"):
    """
    Évolution des sorties de livres (ventes, prêts, dons)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if period == "year":
                cursor.execute("""
                    SELECT 
                        TO_CHAR(transaction_date, 'Mon') as month,
                        EXTRACT(MONTH FROM transaction_date) as month_num,
                        transaction_type,
                        COUNT(*) as count,
                        COALESCE(SUM(quantity), 0) as quantity
                    FROM sorties_livres
                    WHERE transaction_date >= CURRENT_DATE - INTERVAL '12 months'
                    GROUP BY TO_CHAR(transaction_date, 'Mon'), EXTRACT(MONTH FROM transaction_date), transaction_type
                    ORDER BY month_num, transaction_type
                """)
            else:
                cursor.execute("""
                    SELECT 
                        TO_CHAR(transaction_date, 'DD/MM') as day,
                        transaction_type,
                        COUNT(*) as count,
                        COALESCE(SUM(quantity), 0) as quantity
                    FROM sorties_livres
                    WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY TO_CHAR(transaction_date, 'DD/MM'), transaction_date, transaction_type
                    ORDER BY transaction_date, transaction_type
                """)
            
            results = cursor.fetchall()
            cursor.close()
            
            # Organiser les données par type
            labels = sorted(list(set([r['month'] if period == "year" else r['day'] for r in results])))
            
            ventes = []
            prets = []
            dons = []
            
            for label in labels:
                vente_count = sum([r['quantity'] for r in results if (r['month'] if period == "year" else r['day']) == label and r['transaction_type'] == 'vente'])
                pret_count = sum([r['quantity'] for r in results if (r['month'] if period == "year" else r['day']) == label and r['transaction_type'] == 'pret'])
                don_count = sum([r['quantity'] for r in results if (r['month'] if period == "year" else r['day']) == label and r['transaction_type'] == 'don'])
                
                ventes.append(int(vente_count))
                prets.append(int(pret_count))
                dons.append(int(don_count))
            
            return {
                "labels": labels,
                "ventes": ventes,
                "prets": prets,
                "dons": dons
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/revenue-evolution")
async def get_revenue_evolution(period: str = "6months"):
    """
    Évolution des revenus (ventes)
    period: '6months' ou 'year'
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            months = 6 if period == "6months" else 12
            
            cursor.execute(f"""
                SELECT 
                    TO_CHAR(transaction_date, 'Mon') as month,
                    EXTRACT(MONTH FROM transaction_date) as month_num,
                    COALESCE(SUM(sale_price * quantity), 0) as revenue
                FROM sorties_livres
                WHERE transaction_type = 'vente'
                AND transaction_date >= CURRENT_DATE - INTERVAL '{months} months'
                GROUP BY TO_CHAR(transaction_date, 'Mon'), EXTRACT(MONTH FROM transaction_date)
                ORDER BY month_num
            """)
            
            results = cursor.fetchall()
            cursor.close()
            
            return {
                "labels": [r['month'] for r in results],
                "data": [float(r['revenue']) for r in results]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/reading-time")
async def get_reading_time():
    """
    Temps de lecture estimé par jour de la semaine
    Basé sur l'activité (ajouts de citations, résumés, etc.)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Utiliser les citations comme proxy pour l'activité de lecture
            cursor.execute("""
                SELECT 
                    CASE EXTRACT(DOW FROM created_at)
                        WHEN 0 THEN 'Dim'
                        WHEN 1 THEN 'Lun'
                        WHEN 2 THEN 'Mar'
                        WHEN 3 THEN 'Mer'
                        WHEN 4 THEN 'Jeu'
                        WHEN 5 THEN 'Ven'
                        WHEN 6 THEN 'Sam'
                    END as day,
                    EXTRACT(DOW FROM created_at) as day_num,
                    COUNT(*) as activity_count
                FROM citations
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY EXTRACT(DOW FROM created_at)
                ORDER BY day_num
            """)
            
            results = cursor.fetchall()
            cursor.close()
            
            # Convertir l'activité en heures estimées (1 citation = ~15 min de lecture)
            days_order = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
            reading_hours = {day: 0 for day in days_order}
            
            for r in results:
                day = r['day']
                # Estimation: chaque citation représente ~15 min de lecture
                hours = round((int(r['activity_count']) * 15) / 60, 1)
                reading_hours[day] = hours
            
            return {
                "labels": days_order,
                "data": [reading_hours[day] for day in days_order]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/top-books")
async def get_top_books(limit: int = 10):
    """
    Top livres par note
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    id,
                    title,
                    author,
                    rating,
                    category,
                    image_url,
                    price
                FROM books
                WHERE rating IS NOT NULL
                ORDER BY rating DESC, title
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [
                {
                    "id": r['id'],
                    "title": r['title'],
                    "author": r['author'],
                    "rating": float(r['rating']) if r['rating'] else 0,
                    "category": r['category'],
                    "image_url": r['image_url'],
                    "price": float(r['price']) if r['price'] else 0
                }
                for r in results
            ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/recent-activities")
async def get_recent_activities(limit: int = 20):
    """
    Activités récentes (ajouts de livres, sorties, citations, résumés)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            activities = []
            
            # 1. Livres ajoutés récemment
            cursor.execute("""
                SELECT 
                    'book_added' as type,
                    title,
                    author,
                    created_at
                FROM books
                ORDER BY created_at DESC
                LIMIT 10
            """)
            books = cursor.fetchall()
            for b in books:
                activities.append({
                    "type": "book_added",
                    "icon": "fa-book",
                    "color": "#9640e3",
                    "title": f"Livre ajouté: {b['title']}",
                    "description": f"Par {b['author']}",
                    "timestamp": b['created_at'].isoformat() if b['created_at'] else None
                })
            
            # 2. Sorties récentes
            cursor.execute("""
                SELECT 
                    s.transaction_type,
                    s.client_name,
                    s.created_at,
                    b.title
                FROM sorties_livres s
                JOIN books b ON s.book_id = b.id
                ORDER BY s.created_at DESC
                LIMIT 10
            """)
            sorties = cursor.fetchall()
            for s in sorties:
                type_icons = {
                    'vente': ('fa-shopping-cart', '#10b981'),
                    'pret': ('fa-hand-holding-heart', '#3b82f6'),
                    'don': ('fa-gift', '#f59e0b')
                }
                icon, color = type_icons.get(s['transaction_type'], ('fa-exchange', '#666'))
                activities.append({
                    "type": f"sortie_{s['transaction_type']}",
                    "icon": icon,
                    "color": color,
                    "title": f"{s['transaction_type'].capitalize()}: {s['title']}",
                    "description": f"Client: {s['client_name']}",
                    "timestamp": s['created_at'].isoformat() if s['created_at'] else None
                })
            
            # 3. Citations ajoutées
            cursor.execute("""
                SELECT 
                    c.citation_text,
                    c.created_at,
                    b.title
                FROM citations c
                JOIN books b ON c.book_id = b.id
                ORDER BY c.created_at DESC
                LIMIT 10
            """)
            citations = cursor.fetchall()
            for c in citations:
                activities.append({
                    "type": "citation_added",
                    "icon": "fa-quote-right",
                    "color": "#d399fb",
                    "title": f"Citation ajoutée: {c['title']}",
                    "description": c['citation_text'][:50] + "..." if len(c['citation_text']) > 50 else c['citation_text'],
                    "timestamp": c['created_at'].isoformat() if c['created_at'] else None
                })
            
            cursor.close()
            
            # Trier par date et limiter
            activities.sort(key=lambda x: x['timestamp'] if x['timestamp'] else '', reverse=True)
            return activities[:limit]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/inventory-status")
async def get_inventory_status():
    """
    État de l'inventaire (stock disponible, épuisé, etc.)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE copies_count > 5) as high_stock,
                    COUNT(*) FILTER (WHERE copies_count BETWEEN 1 AND 5) as low_stock,
                    COUNT(*) FILTER (WHERE copies_count = 0) as out_of_stock,
                    COUNT(*) as total
                FROM books
            """)
            
            result = cursor.fetchone()
            cursor.close()
            
            return {
                "high_stock": int(result['high_stock'] or 0),
                "low_stock": int(result['low_stock'] or 0),
                "out_of_stock": int(result['out_of_stock'] or 0),
                "total": int(result['total'] or 0)
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/api/dashboard/monthly-comparison")
async def get_monthly_comparison():
    """
    Comparaison mois actuel vs mois précédent
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Livres ajoutés
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE)) as current_month,
                    COUNT(*) FILTER (WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' 
                                     AND created_at < DATE_TRUNC('month', CURRENT_DATE)) as previous_month
                FROM books
            """)
            books_comp = cursor.fetchone()
            
            # Revenus
            cursor.execute("""
                SELECT 
                    COALESCE(SUM(sale_price * quantity) FILTER (WHERE transaction_date >= DATE_TRUNC('month', CURRENT_DATE)), 0) as current_month,
                    COALESCE(SUM(sale_price * quantity) FILTER (WHERE transaction_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' 
                                     AND transaction_date < DATE_TRUNC('month', CURRENT_DATE)), 0) as previous_month
                FROM sorties_livres
                WHERE transaction_type = 'vente'
            """)
            revenue_comp = cursor.fetchone()
            
            # Sorties
            cursor.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE transaction_date >= DATE_TRUNC('month', CURRENT_DATE)) as current_month,
                    COUNT(*) FILTER (WHERE transaction_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' 
                                     AND transaction_date < DATE_TRUNC('month', CURRENT_DATE)) as previous_month
                FROM sorties_livres
            """)
            sorties_comp = cursor.fetchone()
            
            cursor.close()
            
            # Calculer les pourcentages de changement
            def calc_percentage(current, previous):
                if previous == 0:
                    return 100 if current > 0 else 0
                return round(((current - previous) / previous) * 100, 1)
            
            return {
                "books": {
                    "current": int(books_comp['current_month']),
                    "previous": int(books_comp['previous_month']),
                    "percentage": calc_percentage(books_comp['current_month'], books_comp['previous_month'])
                },
                "revenue": {
                    "current": float(revenue_comp['current_month']),
                    "previous": float(revenue_comp['previous_month']),
                    "percentage": calc_percentage(revenue_comp['current_month'], revenue_comp['previous_month'])
                },
                "sorties": {
                    "current": int(sorties_comp['current_month']),
                    "previous": int(sorties_comp['previous_month']),
                    "percentage": calc_percentage(sorties_comp['current_month'], sorties_comp['previous_month'])
                }
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.get("/dashboard.html", response_class=HTMLResponse)
async def read_dashboard_page():
    """Servir la page HTML du dashboard"""
    with open("templates/dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ==================== FIN CODE API DASHBOARD ====================








if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)