#!/usr/bin/env python3
"""
Script de vérification de la base de données - Bibliothèque
Exécutez ce script pour vérifier si la table sorties_livres existe
"""

import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration de la base de données
DB_CONFIG = {
    "host": "localhost",
    "database": "bibliotheque_db",
    "user": "postgres",
    "password": "beauty",
    "port": 5432
}

def print_section(title):
    """Afficher un titre de section"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_table_exists(cursor, table_name):
    """Vérifier si une table existe"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        ) as exists
    """, (table_name,))
    result = cursor.fetchone()
    return result['exists'] if isinstance(result, dict) else result[0]

def check_database():
    """Fonction principale de vérification"""
    try:
        print_section("CONNEXION À LA BASE DE DONNÉES")
        
        # Connexion
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connexion réussie à la base 'bibliotheque_db'")
        
        # Vérifier la table books
        print_section("VÉRIFICATION TABLE BOOKS")
        if check_table_exists(cursor, 'books'):
            cursor.execute("SELECT COUNT(*) as count FROM books")
            result = cursor.fetchone()
            count = result['count'] if isinstance(result, dict) else result[0]
            print(f"✅ Table 'books' existe")
            print(f"📊 Nombre de livres : {count}")
        else:
            print("❌ Table 'books' n'existe pas!")
            print("⚠️  Vous devez d'abord créer la table books (lancez votre serveur FastAPI)")
        
        # Vérifier la table sorties_livres
        print_section("VÉRIFICATION TABLE SORTIES_LIVRES")
        if check_table_exists(cursor, 'sorties_livres'):
            print("✅ Table 'sorties_livres' existe")
            
            # Compter les enregistrements
            cursor.execute("SELECT COUNT(*) as count FROM sorties_livres")
            result = cursor.fetchone()
            count = result['count'] if isinstance(result, dict) else result[0]
            print(f"📊 Nombre de sorties enregistrées : {count}")
            
            # Statistiques par type
            cursor.execute("""
                SELECT 
                    transaction_type,
                    COUNT(*) as nombre,
                    SUM(quantity) as total_quantite
                FROM sorties_livres
                GROUP BY transaction_type
            """)
            stats = cursor.fetchall()
            
            if stats:
                print("\n📈 Répartition par type :")
                for stat in stats:
                    print(f"   - {stat['transaction_type']}: {stat['nombre']} transaction(s), {stat['total_quantite']} exemplaire(s)")
            
            # Vérifier les index
            cursor.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'sorties_livres'
            """)
            indexes = cursor.fetchall()
            print(f"\n🔍 Nombre d'index : {len(indexes)}")
            
        else:
            print("❌ Table 'sorties_livres' n'existe pas!")
            print("\n💡 Pour créer la table, vous avez 2 options :")
            print("   1. Exécuter le script SQL : verification_sorties.sql")
            print("   2. Lancer ce script avec l'option --create")
        
        # Vérifier la colonne copies_count dans books
        print_section("VÉRIFICATION COLONNE COPIES_COUNT")
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'books' AND column_name = 'copies_count'
        """)
        if cursor.fetchone():
            print("✅ Colonne 'copies_count' existe dans la table books")
        else:
            print("❌ Colonne 'copies_count' n'existe pas!")
            print("⚠️  Cette colonne est nécessaire pour gérer le stock")
        
        print_section("RÉSUMÉ")
        print("✅ Vérification terminée avec succès")
        
        cursor.close()
        conn.close()

    except psycopg2.OperationalError as e:
        print_section("ERREUR DE CONNEXION")
        print(f"❌ Impossible de se connecter à la base de données")
        print(f"📝 Erreur : {e}")
        print("\n💡 Vérifiez que :")
        print("   1. PostgreSQL est démarré")
        print("   2. La base 'bibliotheque_db' existe")
        print("   3. Le mot de passe est correct (beauty)")
        
    except Exception as e:
        print_section("ERREUR")
        print(f"❌ Une erreur s'est produite : {e}")

def create_table():
    """Créer la table sorties_livres"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print_section("CRÉATION DE LA TABLE SORTIES_LIVRES")
        
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
                sale_price DECIMAL(10, 2) CHECK (sale_price >= 0),
                payment_method VARCHAR(20) CHECK (payment_method IN ('cash', 'card', 'mobile', 'bank')),
                transaction_date DATE NOT NULL,
                return_date DATE,
                notes TEXT,
                returned BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT valid_vente_price CHECK (
                    transaction_type != 'vente' OR sale_price > 0
                ),
                CONSTRAINT valid_pret_return_date CHECK (
                    transaction_type != 'pret' OR return_date IS NOT NULL
                )
            )
        """)
        print("✅ Table créée")
        
        # Créer les index
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sorties_book_id ON sorties_livres(book_id)",
            "CREATE INDEX IF NOT EXISTS idx_sorties_transaction_type ON sorties_livres(transaction_type)",
            "CREATE INDEX IF NOT EXISTS idx_sorties_transaction_date ON sorties_livres(transaction_date DESC)",
            "CREATE INDEX IF NOT EXISTS idx_sorties_client_phone ON sorties_livres(client_phone)",
            "CREATE INDEX IF NOT EXISTS idx_sorties_returned ON sorties_livres(returned)"
        ]
        
        for index in indexes:
            cursor.execute(index)
        print("✅ Index créés")
        
        # Créer le trigger
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)
        
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_sorties_livres_updated_at ON sorties_livres;
            CREATE TRIGGER update_sorties_livres_updated_at
                BEFORE UPDATE ON sorties_livres
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)
        print("✅ Trigger créé")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n✅ Table 'sorties_livres' créée avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création : {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        create_table()
    else:
        check_database()
        
    print("\n" + "="*60)
    print("  Pour plus d'aide, consultez GUIDE_EXECUTION_SQL.md")
    print("="*60 + "\n")