import datetime
import hashlib
import sqlite3
import datetime
import hashlib
import sqlite3

user_db_file_location = "database_file/users.db"
node_db_file_location = "database_file/nodes.db"
scenario_db_file_location = "database_file/scenarios.db"

def enable_wal_mode(db_file):
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        mode = cursor.fetchone()
        print(f"Journal mode: {mode[0]}")

enable_wal_mode(user_db_file_location)
enable_wal_mode(node_db_file_location)
enable_wal_mode(scenario_db_file_location)

"""
    User Management
"""


def list_users(all_info=False):
    with sqlite3.connect(user_db_file_location) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users")
        result = c.fetchall()

    if not all_info:
        result = [user['user'] for user in result]

    return result


def get_user_info(user):
    with sqlite3.connect(user_db_file_location) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        command = "SELECT * FROM users WHERE user = ?"
        c.execute(command, (user,))
        result = c.fetchone()

    return result


def verify(user, password):
    with sqlite3.connect(user_db_file_location) as conn:
        c = conn.cursor()

        c.execute("SELECT password FROM users WHERE user = ?", (user,))
        result = c.fetchone()
        if result:
            return result[0] == hashlib.sha256(password.encode()).hexdigest()

    return False


def delete_user_from_db(user):
    with sqlite3.connect(user_db_file_location) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE user = ?", (user,))


def add_user(user, password, role):
    with sqlite3.connect(user_db_file_location) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (user.upper(), hashlib.sha256(password.encode()).hexdigest(), role))

def update_user(user, password, role):
    with sqlite3.connect(user_db_file_location) as conn:
        c = conn.cursor()
        print(f"UPDATE users SET password = {hashlib.sha256(password.encode()).hexdigest()}, role = {role} WHERE user = {user.upper()}")
        c.execute("UPDATE users SET password = ?, role = ? WHERE user = ?", (hashlib.sha256(password.encode()).hexdigest(), role, user.upper()))

"""
    Nodes Management
"""


def list_nodes(scenario_name=None, sort_by="idx"):
    # list all nodes in the database
    try:
        with sqlite3.connect(node_db_file_location) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            if scenario_name:
                command = "SELECT * FROM nodes WHERE scenario = ? ORDER BY " + sort_by + ";"
                c.execute(command, (scenario_name,))
            else:
                command = "SELECT * FROM nodes ORDER BY " + sort_by + ";"
                c.execute(command)

            result = c.fetchall()

        return result
    except sqlite3.Error as e:
        print(f"Error occurred while listing nodes: {e}")
        return None


def list_nodes_by_scenario_name(scenario_name):
    try:
        with sqlite3.connect(node_db_file_location) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            command = "SELECT * FROM nodes WHERE scenario = ? ORDER BY idx;"
            c.execute(command, (scenario_name,))
            result = c.fetchall()

        return result
    except sqlite3.Error as e:
        print(f"Error occurred while listing nodes by scenario name: {e}")
        return None


def get_location_neighbors(node_uid, scenario_name):
    try:
        with sqlite3.connect(node_db_file_location) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            # Obtener los vecinos del nodo
            command = "SELECT neighbors FROM nodes WHERE uid = ? AND scenario = ?;"
            c.execute(command, (node_uid, scenario_name))
            result = c.fetchone()

            if not result or not result["neighbors"]:
                return []

            neighbors_list = result["neighbors"].split(" ")

            # Obtener la ubicación de los vecinos
            ip_port_pairs = [(node.split(":")[0], node.split(":")[1]) for node in neighbors_list if ":" in node]
            placeholders = ", ".join(["(?, ?)" for _ in ip_port_pairs])
            command = f"SELECT ip, port, latitude, longitude FROM nodes WHERE scenario = ? AND (ip, port) IN ({placeholders});"
            
            params = [scenario_name] + [item for sublist in ip_port_pairs for item in sublist]
            c.execute(command, params)
            result = c.fetchall()

            # Procesar los resultados
            neighbors = {f"{node['ip']}:{node['port']}": [node["latitude"], node["longitude"]] for node in result}

            return neighbors
    except sqlite3.Error as e:
        print(f"Error occurred while getting location neighbors: {e}")
        return []



def update_node_record(node_uid, idx, ip, port, role, neighbors, latitude, longitude, timestamp, federation, federation_round, scenario):
    # Check if the node record with node_uid and scenario already exists in the database
    # If it does, update the record
    # If it does not, create a new record
    _conn = sqlite3.connect(node_db_file_location)
    _c = _conn.cursor()

    command = "SELECT * FROM nodes WHERE uid = ? AND scenario = ?;"
    _c.execute(command, (node_uid, scenario))
    result = _c.fetchone()

    if result is None:
        # Create a new record
        _c.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (node_uid, idx, ip, port, role, neighbors, latitude, longitude, timestamp, federation, federation_round, scenario))
    else:
        # Update the record
        command = "UPDATE nodes SET idx = ?, ip = ?, port = ?, role = ?, neighbors = ?, latitude = ?, longitude = ?, timestamp = ?, federation = ?, round = ? WHERE uid = ? AND scenario = ?;"
        _c.execute(command, (idx, ip, port, role, neighbors, latitude, longitude, timestamp, federation, federation_round, node_uid, scenario))

    _conn.commit()
    _conn.close()


def remove_all_nodes():
    with sqlite3.connect(node_db_file_location) as conn:
        c = conn.cursor()
        command = "DELETE FROM nodes;"
        c.execute(command)
        conn.commit()


def remove_nodes_by_scenario_name(scenario_name):
    with sqlite3.connect(node_db_file_location) as conn:
        c = conn.cursor()
        command = "DELETE FROM nodes WHERE scenario = ?;"
        c.execute(command, (scenario_name,))
        conn.commit()


"""
    Scenario Management
"""


def get_all_scenarios(sort_by="start_time"):
    with sqlite3.connect(scenario_db_file_location) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        command = "SELECT * FROM scenarios ORDER BY ?;"
        c.execute(command, (sort_by,))
        result = c.fetchall()

    return result


def get_all_scenarios_and_check_completed(sort_by="start_time"):
    with sqlite3.connect(scenario_db_file_location) as _conn:
        _conn.row_factory = sqlite3.Row
        _c = _conn.cursor()
        command = "SELECT * FROM scenarios ORDER BY ?;"
        _c.execute(command, (sort_by,))
        result = _c.fetchall()

        for scenario in result:
            if scenario['status'] == "running":
                if check_scenario_federation_completed(scenario['name']):
                    scenario_set_status_to_completed(scenario['name'])
                    result = get_all_scenarios()

    return result


def scenario_update_record(scenario_name, start_time, end_time, title, description, status, network_subnet, model, dataset, rounds, role):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "SELECT * FROM scenarios WHERE name = ?;"
    _c.execute(command, (scenario_name,))
    result = _c.fetchone()

    if result is None:
        # Create a new record
        _c.execute("INSERT INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (scenario_name, start_time, end_time, title, description, status, network_subnet, model, dataset, rounds, role))
    else:
        # Update the record
        command = "UPDATE scenarios SET start_time = ?, end_time = ?, title = ?, description = ?, status = ?, network_subnet = ?, model = ?, dataset = ?, rounds = ?, role = ? WHERE name = ?;"
        _c.execute(command, (start_time, end_time, title, description, status, network_subnet, model, dataset, rounds, role, scenario_name))

    _conn.commit()
    _conn.close()


def scenario_set_all_status_to_finished():
    # Set all running scenarios to finished and update the end_time to the current time
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "UPDATE scenarios SET status = 'finished', end_time = ? WHERE status = 'running';"
    current_time = str(datetime.datetime.now())
    _c.execute(command, (current_time,))

    _conn.commit()
    _conn.close()


def scenario_set_status_to_finished(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "UPDATE scenarios SET status = 'finished', end_time = ? WHERE name = ?;"
    current_time = str(datetime.datetime.now())
    _c.execute(command, (current_time, scenario_name))

    _conn.commit()
    _conn.close()


def scenario_set_status_to_completed(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "UPDATE scenarios SET status = 'completed' WHERE name = ?;"
    _c.execute(command, (scenario_name,))

    _conn.commit()
    _conn.close()


def get_running_scenario():
    with sqlite3.connect(scenario_db_file_location) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        command = "SELECT * FROM scenarios WHERE status = ? OR status = ?;"
        c.execute(command, ('running', 'completed'))
        result = c.fetchone()

    return result


def get_completed_scenario():
    with sqlite3.connect(scenario_db_file_location) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        command = "SELECT * FROM scenarios WHERE status = ?;"
        c.execute(command, ('completed',))
        result = c.fetchone()

    return result

def get_scenario_by_name(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()
    command = "SELECT * FROM scenarios WHERE name = ?;"
    _c.execute(command, (scenario_name,))
    result = _c.fetchone()

    _conn.commit()
    _conn.close()

    return result


def remove_scenario_by_name(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "DELETE FROM scenarios WHERE name = ?;"
    _c.execute(command, (scenario_name,))

    _conn.commit()
    _conn.close()


def check_scenario_federation_completed(scenario_name):
    try:
        # Connect to the scenario database to get the total rounds for the scenario
        with sqlite3.connect(scenario_db_file_location) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            c.execute("SELECT rounds FROM scenarios WHERE name = ?;", (scenario_name,))
            scenario = c.fetchone()

            if not scenario:
                raise ValueError(f"Scenario '{scenario_name}' not found.")

            total_rounds = scenario['rounds']

        # Connect to the node database to check the rounds for each node
        with sqlite3.connect(node_db_file_location) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            c.execute("SELECT round FROM nodes WHERE scenario = ?;", (scenario_name,))
            nodes = c.fetchall()

            if len(nodes) == 0:
                return False

            # Check if all nodes have completed the total rounds
            return all(node['round'] == total_rounds for node in nodes)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def check_scenario_with_role(role, scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()
    command = "SELECT * FROM scenarios WHERE role = ? AND name = ?;"
    _c.execute(command, (role, scenario_name))
    result = _c.fetchone()

    _conn.commit()
    _conn.close()

    return result


if __name__ == "__main__":
    print(list_users())
