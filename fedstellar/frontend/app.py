import argparse
import datetime
import hashlib
import io
import json
import os
import signal
import sys
import zipfile
from urllib.parse import urlencode

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the path two directories up to the system path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from ansi2html import Ansi2HTMLConverter

from flask import (
    Flask,
    session,
    url_for,
    redirect,
    render_template,
    request,
    abort,
    flash,
    send_file,
    make_response,
    jsonify,
    Response,
)
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from fedstellar.frontend.database import (
    list_users,
    verify,
    delete_user_from_db,
    add_user,
    update_user,
    scenario_update_record,
    scenario_set_all_status_to_finished,
    get_running_scenario,
    get_user_info,
    get_scenario_by_name,
    list_nodes_by_scenario_name,
    remove_nodes_by_scenario_name,
    remove_scenario_by_name,
    scenario_set_status_to_finished,
    get_all_scenarios_and_check_completed,
    check_scenario_with_role,
    get_location_neighbors,
    update_node_record,
)

import eventlet

eventlet.monkey_patch()
async_mode = "eventlet"

app = Flask(__name__)
app.config.from_object("config")
app.config["log_dir"] = os.environ.get("FEDSTELLAR_LOGS_DIR")
app.config["config_dir"] = os.environ.get("FEDSTELLAR_CONFIG_DIR")
app.config["model_dir"] = os.environ.get("FEDSTELLAR_MODELS_DIR")
app.config["root_host_path"] = os.environ.get("FEDSTELLAR_ROOT_HOST")
socketio = SocketIO(
    app,
    async_mode=async_mode,
    logger=False,
    engineio_logger=False,
    cors_allowed_origins="*",
)


# Detect CTRL+C from parent process
def signal_handler(signal, frame):
    print("You pressed Ctrl+C [frontend]!")
    scenario_set_all_status_to_finished()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


@app.errorhandler(401)
def fedstellar_401(error):
    return render_template("401.html"), 401


@app.errorhandler(403)
def fedstellar_403(error):
    return render_template("403.html"), 403


@app.errorhandler(404)
def fedstellar_404(error):
    return render_template("404.html"), 404


@app.errorhandler(405)
def fedstellar_405(error):
    return render_template("405.html"), 405


@app.errorhandler(413)
def fedstellar_413(error):
    return render_template("413.html"), 413


@app.template_filter("datetimeformat")
def datetimeformat(value, format="%B %d, %Y %H:%M"):
    return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime(format)


@app.route("/")
def fedstellar_home():
    # Get alerts and news from API
    import requests

    alerts = []
    # # Use custom headers
    # headers = {"User-Agent": "Fedstellar Frontend"}
    # url = "https://federatedlearning.inf.um.es/alerts/alerts"
    # try:
    #     response = requests.get(url, headers=headers)
    #     alerts = response.json()
    # except requests.exceptions.RequestException as e:
    #     print(e)
    #     alerts = []
    return render_template("index.html", alerts=alerts)


@app.route("/scenario/<scenario_name>/private/")
def fedstellar_scenario_private(scenario_name):
    if "user" in session.keys():
        return render_template(
            "private.html",
            scenario_name=scenario_name,
        )
    else:
        return abort(401)


@app.route("/admin/")
def fedstellar_admin():
    if session.get("role") == "admin":
        user_list = list_users(all_info=True)
        user_table = zip(
            range(1, len(user_list) + 1),
            [user[0] for user in user_list],
            [user[2] for user in user_list]
        )
        return render_template("admin.html", users=user_table)
    else:
        return abort(401)


def send_from_directory(directory, filename, **options):
    """Sends a file from a given directory with :func:`send_file`.

    :param directory: the directory to look for the file in.
    :param filename: the name of the file to send.
    :param options: the options to forward to :func:`send_file`.
    """
    return send_file(os.path.join(directory, filename), **options)


@app.route("/login", methods=["POST"])
def fedstellar_login():
    user_submitted = request.form.get("user").upper()
    if (user_submitted in list_users()) and verify(
        user_submitted, request.form.get("password")
    ):
        user_info = get_user_info(user_submitted)
        session["user"] = user_submitted
        session["role"] = user_info[2]
        return "Login successful", 200
    else:
        # flash(u'Invalid password provided', 'error')
        abort(401)


@app.route("/logout/")
def fedstellar_logout():
    session.pop("user", None)
    return redirect(url_for("fedstellar_home"))


@app.route("/user/delete/<user>/", methods=["GET"])
def fedstellar_delete_user(user):
    if session.get("role", None) == "admin":
        if user == "ADMIN":  # ADMIN account can't be deleted.
            return abort(403)
        if user == session["user"]:  # Current user can't delete himself.
            return abort(403)

        delete_user_from_db(user)
        return redirect(url_for("fedstellar_admin"))
    else:
        return abort(401)


@app.route("/user/add", methods=["POST"])
def fedstellar_add_user():
    if session.get("role", None) == "admin":  # only Admin should be able to add user.
        # before we add the user, we need to ensure this doesn't exit in database. We also need to ensure the id is valid.
        user_list = list_users(all_info=True)
        if request.form.get("user").upper() in user_list:
            return redirect(url_for("fedstellar_admin"))
        elif (
            " " in request.form.get("user")
            or "'" in request.form.get("user")
            or '"' in request.form.get("user")
        ):
            return redirect(url_for("fedstellar_admin"))
        else:
            add_user(
                request.form.get("user"),
                request.form.get("password"),
                request.form.get("role"),
            )
            return redirect(url_for("fedstellar_admin"))
    else:
        return abort(401)


@app.route("/user/update", methods=["POST"])
def fedstellar_update_user():
    if session.get("role", None) == "admin":
        user = request.form.get("user")
        password = request.form.get("password")
        role = request.form.get("role")
        
        user_list = list_users()
        if user not in user_list:
            return redirect(url_for("fedstellar_admin"))
        else:
            update_user(user, password, role)
            return redirect(url_for("fedstellar_admin"))
    else:
        return abort(401)

#                                                   #
# -------------- Scenario management -------------- #
#                                                   #
@app.route("/api/scenario/", methods=["GET"])
@app.route("/scenario/", methods=["GET"])
def fedstellar_scenario():
    if "user" in session.keys() or request.path == "/api/scenario/":
        # Get the list of scenarios
        scenarios = (
            get_all_scenarios_and_check_completed()
        )  # Get all scenarios after checking if they are completed
        scenario_running = get_running_scenario()
        # Check if status of scenario_running is "completed"
        bool_completed = False
        if scenario_running:
            bool_completed = scenario_running[5] == "completed"
        if scenarios:
            if request.path == "/scenario/":
                return render_template(
                    "scenario.html",
                    scenarios=scenarios,
                    scenario_running=scenario_running,
                    scenario_completed=bool_completed,
                )
            elif request.path == "/api/scenario/":
                return jsonify(scenarios), 200
            else:
                return abort(401)

        else:
            if request.path == "/scenario/":
                return render_template("scenario.html")
            elif request.path == "/api/scenario/":
                return jsonify({"scenarios_status": "not found in database"}), 200
            else:
                return abort(401)
    else:
        return abort(401)


#                                                   #
# ------------------- Monitoring ------------------ #
#                                                   #


@app.route("/api/scenario/<scenario_name>/monitoring", methods=["GET"])
@app.route("/scenario/<scenario_name>/monitoring", methods=["GET"])
def fedstellar_scenario_monitoring(scenario_name):
    if "user" in session.keys():
        scenario = get_scenario_by_name(scenario_name)
        if scenario:
            nodes_list = list_nodes_by_scenario_name(scenario_name)
            if nodes_list:
                # Get json data from each node configuration file
                nodes_config = []
                # Generate an array with True for each node that is running
                nodes_status = []
                nodes_offline = []
                for i, node in enumerate(nodes_list):
                    nodes_config.append((node[2], node[3], node[4]))  # IP, Port, Role
                    if datetime.datetime.now() - datetime.datetime.strptime(
                        node[8], "%Y-%m-%d %H:%M:%S.%f"
                    ) > datetime.timedelta(seconds=20):
                        nodes_status.append(False)
                        nodes_offline.append(node[2] + ":" + str(node[3]))
                    else:
                        nodes_status.append(True)
                nodes_table = zip(
                    [x[0] for x in nodes_list],  # UID
                    [x[1] for x in nodes_list],  # IDX
                    [x[2] for x in nodes_list],  # IP
                    [x[3] for x in nodes_list],  # Port
                    [x[4] for x in nodes_list],  # Role
                    [x[5] for x in nodes_list],  # Neighbors
                    [x[6] for x in nodes_list],  # Latitude
                    [x[7] for x in nodes_list],  # Longitude
                    [x[8] for x in nodes_list],  # Timestamp
                    [x[9] for x in nodes_list],  # Federation
                    [x[10] for x in nodes_list],  # Round
                    [x[11] for x in nodes_list],  # Scenario name
                    nodes_status,  # Status
                )

                if os.path.exists(
                    os.path.join(
                        app.config["config_dir"], scenario_name, "topology.png"
                    )
                ):
                    if os.path.getmtime(
                        os.path.join(
                            app.config["config_dir"], scenario_name, "topology.png"
                        )
                    ) < max(
                        [
                            os.path.getmtime(
                                os.path.join(
                                    app.config["config_dir"],
                                    scenario_name,
                                    f"participant_{node[1]}.json",
                                )
                            )
                            for node in nodes_list
                        ]
                    ):
                        # Update the 3D topology and image
                        update_topology(scenario[0], nodes_list, nodes_config)
                else:
                    update_topology(scenario[0], nodes_list, nodes_config)

                if request.path == "/scenario/" + scenario_name + "/monitoring":
                    return render_template(
                        "monitoring.html",
                        scenario_name=scenario_name,
                        scenario=scenario,
                        nodes=nodes_table,
                    )
                elif request.path == "/api/scenario/" + scenario_name + "/monitoring":
                    return (
                        jsonify(
                            {
                                "scenario_status": scenario[5],
                                "nodes_table": list(nodes_table),
                                "scenario_name": scenario[0],
                                "scenario_title": scenario[3],
                                "scenario_description": scenario[4],
                            }
                        ),
                        200,
                    )
                else:
                    return abort(401)
            else:
                # There is a scenario but no nodes
                if request.path == "/scenario/" + scenario_name + "/monitoring":
                    return render_template(
                        "monitoring.html",
                        scenario_name=scenario_name,
                        scenario=scenario,
                        nodes=[],
                    )
                elif request.path == "/api/scenario/" + scenario_name + "/monitoring":
                    return (
                        jsonify(
                            {
                                "scenario_status": scenario[5],
                                "nodes_table": [],
                                "scenario_name": scenario[0],
                                "scenario_title": scenario[3],
                                "scenario_description": scenario[4],
                            }
                        ),
                        200,
                    )
                else:
                    return abort(401)
        else:
            # There is no scenario
            if request.path == "/scenario/" + scenario_name + "/monitoring":
                return render_template(
                    "monitoring.html",
                    scenario_name=scenario_name,
                    scenario=None,
                    nodes=[],
                )
            elif request.path == "/api/scenario/" + scenario_name + "/monitoring":
                return jsonify({"scenario_status": "not exists"}), 200
            else:
                return abort(401)
    else:
        return abort(401)


def update_topology(scenario_name, nodes_list, nodes_config):
    import numpy as np

    nodes = []
    for node in nodes_list:
        nodes.append(node[2] + ":" + str(node[3]))
    matrix = np.zeros((len(nodes), len(nodes)))
    for node in nodes_list:
        for neighbour in node[5].split(" "):
            if neighbour != "":
                if neighbour in nodes:
                    matrix[
                        nodes.index(node[2] + ":" + str(node[3])),
                        nodes.index(neighbour),
                    ] = 1
    from fedstellar.utils.topologymanager import TopologyManager

    tm = TopologyManager(
        n_nodes=len(nodes_list), topology=matrix, scenario_name=scenario_name
    )
    tm.update_nodes(nodes_config)
    tm.draw_graph(
        path=os.path.join(app.config["config_dir"], scenario_name, f"topology.png")
    )  # TODO: Improve this


@app.route("/scenario/<scenario_name>/node/update", methods=["POST"])
def fedstellar_update_node(scenario_name):
    if request.method == "POST":
        # Check if the post request is a json, if not, return 400
        if request.is_json:
            config = request.get_json()
            timestamp = datetime.datetime.now()
            # Update file in the local directory (not needed, reduce overhead)
            # with open(
            #     os.path.join(
            #         app.config["config_dir"],
            #         scenario_name,
            #         f'participant_{config["device_args"]["idx"]}.json',
            #     ),
            #     "w",
            # ) as f:
            #     json.dump(config, f, sort_keys=False, indent=2)

            # Update the node in database
            update_node_record(
                str(config["device_args"]["uid"]),
                str(config["device_args"]["idx"]),
                str(config["network_args"]["ip"]),
                str(config["network_args"]["port"]),
                str(config["device_args"]["role"]),
                str(config["network_args"]["neighbors"]),
                str(config["mobility_args"]["latitude"]),
                str(config["mobility_args"]["longitude"]),
                str(timestamp),
                str(config["scenario_args"]["federation"]),
                str(config["federation_args"]["round"]),
                str(config["scenario_args"]["name"]),
            )

            # Update the mobility file with the geographical distance and the neighbours
            from geopy import distance

            neigbours_location = get_location_neighbors(
                str(config["device_args"]["uid"]), str(config["scenario_args"]["name"])
            )  # {neighbour: [latitude, longitude]}

            if neigbours_location:
                for neighbour in neigbours_location:
                    neigbours_location[neighbour].append(
                        distance.distance(
                            (
                                config["mobility_args"]["latitude"],
                                config["mobility_args"]["longitude"],
                            ),
                            (
                                neigbours_location[neighbour][0],
                                neigbours_location[neighbour][1],
                            ),
                        ).m
                    )

            with open(
                os.path.join(
                    app.config["log_dir"],
                    scenario_name,
                    f'participant_{config["device_args"]["idx"]}_mobility.csv',
                ),
                "a+",
            ) as f:
                if (
                    os.stat(
                        os.path.join(
                            app.config["log_dir"],
                            scenario_name,
                            f'participant_{config["device_args"]["idx"]}_mobility.csv',
                        )
                    ).st_size
                    == 0
                ):
                    f.write("timestamp,round,neighbor,latitude,longitude,distance\n")

                #f.write(
                #    f"{timestamp},{str(config['federation_args']['round'])},{config['network_args']['ip'] + ':' + str(config['network_args']['port'])},{config['mobility_args']['latitude']},{config['mobility_args']['longitude']},None\n"
                #)
                
                for neighbour in config["network_args"]["neighbors"].split(" "):
                    if neighbour != "":
                        try:
                            f.write(
                                f"{timestamp},{str(config['federation_args']['round'])},{neighbour},{neigbours_location[neighbour][0]},{neigbours_location[neighbour][1]},{neigbours_location[neighbour][2]}\n"
                            )
                        except:
                            f.write(f"{timestamp},{str(config['federation_args']['round'])},None,None,{neighbour},None\n")
                            
            node_update = {
                    "scenario_name": scenario_name,
                    "uid": config["device_args"]["uid"],
                    "idx": config["device_args"]["idx"],
                    "ip": config["network_args"]["ip"],
                    "port": str(config["network_args"]["port"]),
                    "role": config["device_args"]["role"],
                    "neighbors": config["network_args"]["neighbors"],
                    "latitude": config["mobility_args"]["latitude"],
                    "longitude": config["mobility_args"]["longitude"],
                    "timestamp": str(timestamp),
                    "federation": config["scenario_args"]["federation"],
                    "round": config["federation_args"]["round"],
                    "name": config["scenario_args"]["name"],
                    "status": True,
                    "neigbours_location": neigbours_location,
            }

            # Send notification to each connected users
            socketio.emit("node_update", node_update)

            return make_response(jsonify(node_update), 200)
        else:
            return abort(400)


@app.route("/scenario/<scenario_name>/node/<id>/infolog", methods=["GET"])
def fedstellar_monitoring_log(scenario_name, id):
    if "user" in session.keys():
        logs = os.path.join(
            app.config["log_dir"], scenario_name, f"participant_{id}.log"
        )
        if os.path.exists(logs):
            return send_file(logs, mimetype="text/plain", as_attachment=True)
        else:
            abort(404)
    else:
        make_response("You are not authorized to access this page.", 401)


@app.route("/scenario/<scenario_name>/node/<id>/infolog/<number>", methods=["GET"])
def fedstellar_monitoring_log_x(scenario_name, id, number):
    if "user" in session.keys():
        # Send file (is not a json file) with the log
        logs = os.path.join(
            app.config["log_dir"], scenario_name, f"participant_{id}.log"
        )
        if os.path.exists(logs):
            # Open file maintaining the file format (for example, new lines)
            with open(logs, "r") as f:
                # Read the last n lines of the file
                lines = f.readlines()[-int(number) :]
                # Join the lines in a single string
                lines = "".join(lines)
                # Convert the ANSI escape codes to HTML
                converter = Ansi2HTMLConverter()
                html_text = converter.convert(lines, full=False)
                # Return the string
                return Response(html_text, mimetype="text/plain")
        else:
            return Response("No logs available", mimetype="text/plain")

    else:
        make_response("You are not authorized to access this page.", 401)


@app.route("/scenario/<scenario_name>/node/<id>/debuglog", methods=["GET"])
def fedstellar_monitoring_log_debug(scenario_name, id):
    if "user" in session.keys():
        logs = os.path.join(
            app.config["log_dir"], scenario_name, f"participant_{id}_debug.log"
        )
        if os.path.exists(logs):
            return send_file(logs, mimetype="text/plain", as_attachment=True)
        else:
            abort(404)
    else:
        make_response("You are not authorized to access this page.", 401)


@app.route("/scenario/<scenario_name>/node/<id>/errorlog", methods=["GET"])
def fedstellar_monitoring_log_error(scenario_name, id):
    if "user" in session.keys():
        logs = os.path.join(
            app.config["log_dir"], scenario_name, f"participant_{id}_error.log"
        )
        if os.path.exists(logs):
            return send_file(logs, mimetype="text/plain", as_attachment=True)
        else:
            abort(404)
    else:
        make_response("You are not authorized to access this page.", 401)


@app.route(
    "/scenario/<scenario_name>/topology/image/", methods=["GET"]
)  # TODO: maybe change scenario name and save directly in config folder
def fedstellar_monitoring_image(scenario_name):
    if "user" in session.keys():
        topology_image = os.path.join(
            app.config["config_dir"], scenario_name, f"topology.png"
        )
        if os.path.exists(topology_image):
            return send_file(topology_image, mimetype="image/png")
        else:
            abort(404)
    else:
        make_response("You are not authorized to access this page.", 401)


def stop_scenario(scenario_name):
    from fedstellar.controller import Controller

    Controller.stop_participants()
    scenario_set_status_to_finished(scenario_name)


def stop_all_scenarios():
    from fedstellar.controller import Controller

    Controller.stop_participants()
    scenario_set_all_status_to_finished()


@app.route("/scenario/<scenario_name>/stop", methods=["GET"])
def fedstellar_stop_scenario(scenario_name):
    # Stop the scenario
    if "user" in session.keys():
        if session["role"] == "demo":
            return abort(401)
        elif session["role"] == "user":
            if not check_scenario_with_role(session["role"], scenario_name):
                return abort(401)
        stop_scenario(scenario_name)
        return redirect(url_for("fedstellar_scenario"))
    else:
        return abort(401)


def remove_scenario(scenario_name=None):
    from fedstellar.controller import Controller

    remove_nodes_by_scenario_name(scenario_name)
    remove_scenario_by_name(scenario_name)
    Controller.remove_files_by_scenario(scenario_name)


@app.route("/scenario/<scenario_name>/remove", methods=["GET"])
def fedstellar_remove_scenario(scenario_name):
    # Remove the scenario
    if "user" in session.keys():
        if session["role"] == "demo":
            return abort(401)
        elif session["role"] == "user":
            if not check_scenario_with_role(session["role"], scenario_name):
                return abort(401)
        remove_scenario(scenario_name)
        return redirect(url_for("fedstellar_scenario"))
    else:
        return abort(401)


#                                                   #
# ------------------- Statistics ------------------ #
#                                                   #


@app.route("/scenario/statistics/", methods=["GET"])
@app.route("/scenario/<scenario_name>/statistics/", methods=["GET"])
def fedstellar_scenario_statistics(scenario_name=None):
    if "user" in session.keys():
        # Adjust the filter to the scenario name
        tensorboard_url = "/statistics/"
        if scenario_name is not None:
            tensorboard_url += f"?runFilter={scenario_name}"

        return render_template("statistics.html", tensorboard_url=tensorboard_url)
    else:
        return abort(401)


@app.route("/statistics/", methods=["GET", "POST"])
@app.route("/statistics/<path:path>", methods=["GET", "POST"])
def statistics_proxy(path=None):
    if "user" in session.keys():
        import requests

        query_string = urlencode(request.args)

        url = f"http://localhost:8080"

        tensorboard_url = f"{url}{('/' + path) if path else ''}" + (
            "?" + query_string if query_string else ""
        )

        response = requests.request(
            method=request.method,
            url=tensorboard_url,
            headers={key: value for (key, value) in request.headers if key != "Host"},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
        )

        excluded_headers = [
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "connection",
        ]
        headers = [
            (name, value)
            for (name, value) in response.raw.headers.items()
            if name.lower() not in excluded_headers
        ]

        if "text/html" in response.headers["Content-Type"]:
            # Replace the resources URLs to point to the proxy
            content = response.text
            content = content.replace("url(/", f"url(/statistics/")
            response = Response(content, response.status_code, headers)
            return response

        # Construye y envía la respuesta
        response = Response(response.content, response.status_code, headers)
        return response

    else:
        return abort(401)


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


@app.route("/scenario/<scenario_name>/download", methods=["GET"])
def fedstellar_scenario_download(scenario_name):
    if "user" in session.keys():
        log_folder = os.path.join(app.config["log_dir"], scenario_name)
        config_folder = os.path.join(app.config["config_dir"], scenario_name)
        if os.path.exists(log_folder) and os.path.exists(config_folder):
            # Create a zip file with the logs and the config files, send it to the user and delete it
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipdir(log_folder, zipf)
                zipdir(config_folder, zipf)

            memory_file.seek(0)

            return send_file(
                memory_file,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"{scenario_name}.zip",
            )
    else:
        return abort(401)


#                                                   #
# ------------------- Deployment ------------------ #
#                                                   #


@app.route("/scenario/deployment/", methods=["GET"])
def fedstellar_scenario_deployment():
    if "user" in session.keys():
        scenario_running = get_running_scenario()
        return render_template("deployment.html", scenario_running=scenario_running)
    else:
        return abort(401)


def attack_node_assign(
    nodes,
    federation,
    attack,
    poisoned_node_percent,
    poisoned_sample_percent,
    poisoned_noise_percent,
):
    """Identify which nodes will be attacked"""
    import random
    import math

    attack_matrix = []
    n_nodes = len(nodes)
    if n_nodes == 0:
        return attack_matrix

    nodes_index = []
    # Get the nodes index
    if federation == "DFL":
        nodes_index = list(nodes.keys())
    else:
        for node in nodes:
            if nodes[node]["role"] != "server":
                nodes_index.append(node)

    n_nodes = len(nodes_index)
    # Number of attacked nodes, round up
    num_attacked = int(math.ceil(poisoned_node_percent / 100 * n_nodes))
    if num_attacked > n_nodes:
        num_attacked = n_nodes

    # Get the index of attacked nodes
    attacked_nodes = random.sample(nodes_index, num_attacked)

    # Assign the role of each node
    for node in nodes:
        node_att = "No Attack"
        attack_sample_persent = 0
        poisoned_ratio = 0
        if (node in attacked_nodes) or (nodes[node]["malicious"]):
            node_att = attack
            attack_sample_persent = poisoned_sample_percent / 100
            poisoned_ratio = poisoned_noise_percent / 100
        nodes[node]["attacks"] = node_att
        nodes[node]["poisoned_sample_percent"] = attack_sample_persent
        nodes[node]["poisoned_ratio"] = poisoned_ratio
        attack_matrix.append([node, node_att, attack_sample_persent, poisoned_ratio])
    return nodes, attack_matrix

import math

def mobility_assign(nodes, mobile_participants_percent):
    """Assign mobility to nodes"""
    import random

    # Number of mobile nodes, round down
    num_mobile = math.floor(mobile_participants_percent / 100 * len(nodes))
    if num_mobile > len(nodes):
        num_mobile = len(nodes)

    # Get the index of mobile nodes
    mobile_nodes = random.sample(list(nodes.keys()), num_mobile)

    # Assign the role of each node
    for node in nodes:
        node_mob = False
        if node in mobile_nodes:
            node_mob = True
        nodes[node]["mobility"] = node_mob
    return nodes

@app.route("/scenario/deployment/run", methods=["POST"])
def fedstellar_scenario_deployment_run():
    from fedstellar.controller import Controller

    if "user" in session.keys():
        if session["role"] == "demo":
            return abort(401)
        elif session["role"] == "user":
            # If there is a scenario running, abort
            if get_running_scenario():
                return abort(401)
        # Receive a JSON data with the scenario configuration
        if request.is_json:
            # Stop the running scenario
            stop_all_scenarios()
            data = request.get_json()
            nodes = data["nodes"]
            scenario_name = f'fedstellar_{data["federation"]}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}'
            
            scenario_path = os.path.join(app.config["config_dir"], scenario_name)
            os.makedirs(scenario_path, exist_ok=True)

            scenario_file = os.path.join(scenario_path, "scenario.json")
            with open(scenario_file, "w") as f:
                json.dump(data, f, sort_keys=False, indent=2)

            args_controller = {
                "scenario_name": scenario_name,
                "config": app.config["config_dir"],
                "logs": app.config["log_dir"],
                "models": app.config["model_dir"],
                "n_nodes": data["n_nodes"],
                "matrix": data["matrix"],
                "federation": data["federation"],
                "topology": data["topology"],
                "simulation": data["simulation"],
                "env": None,
                "root_path": app.config["root_host_path"],
                "webport": request.host.split(":")[1]
                if ":" in request.host
                else 80,  # Get the port of the frontend, if not specified, use 80
                "network_subnet": data["network_subnet"],
                "network_gateway": data["network_gateway"],
            }
            # Save args in a file
            controller_file = os.path.join(
                app.config["config_dir"], scenario_name, "controller.json"
            )
            with open(controller_file, "w") as f:
                json.dump(args_controller, f, sort_keys=False, indent=2)
            
            
            # Get attack info
            attack = data["attacks"]
            poisoned_node_percent = int(data["poisoned_node_percent"])
            poisoned_sample_percent = int(data["poisoned_sample_percent"])
            poisoned_noise_percent = int(data["poisoned_noise_percent"])
            federation = data["federation"]
            # Get attack matrix
            nodes, attack_matrix = attack_node_assign(
                nodes,
                federation,
                attack,
                poisoned_node_percent,
                poisoned_sample_percent,
                poisoned_noise_percent,
            )
            
            
            mobility_status = data["mobility"]
            if mobility_status:
                # Mobility parameters (selecting mobile nodes)
                mobile_participants_percent = int(data["mobile_participants_percent"])
                # Assign mobility to nodes depending on the percentage
                nodes = mobility_assign(nodes, mobile_participants_percent)
            else:
                # Assign mobility to nodes depending on the percentage
                nodes = mobility_assign(nodes, 0)

            # For each node, create a new file in config directory
            import shutil
            # Loop dictionary of nodes
            for node in nodes:
                node_config = nodes[node]
                # Create a copy of participant.json.example and update the file with the update values
                participant_file = os.path.join(
                    app.config["config_dir"],
                    scenario_name,
                    f'participant_{node_config["id"]}.json',
                )
                os.makedirs(os.path.dirname(participant_file), exist_ok=True)
                # Create a copy of participant.json.example
                shutil.copy(
                    os.path.join(
                        app.config["CONFIG_FOLDER_FRONTEND"],
                        f"participant.json.example",
                    ),
                    participant_file,
                )
                # Update IP, port, and role
                with open(participant_file) as f:
                    participant_config = json.load(f)
                participant_config["network_args"]["ip"] = node_config["ip"]
                participant_config["network_args"]["port"] = int(node_config["port"])
                participant_config["device_args"]["idx"] = node_config["id"]
                participant_config["device_args"]["start"] = node_config["start"]
                participant_config["device_args"]["role"] = node_config["role"]
                participant_config["device_args"]["malicious"] = node_config[
                    "malicious"
                ]
                participant_config["scenario_args"]["rounds"] = int(data["rounds"])
                participant_config["data_args"]["dataset"] = data["dataset"]
                participant_config["data_args"]["iid"] = data["iid"]
                participant_config["model_args"]["model"] = data["model"]
                participant_config["training_args"]["epochs"] = int(data["epochs"])
                participant_config["device_args"]["accelerator"] = data[
                    "accelerator"
                ]
                participant_config["device_args"]["logging"] = data[
                    "logginglevel"
                ]
                participant_config["aggregator_args"]["algorithm"] = data["agg_algorithm"]
                
                participant_config["adversarial_args"]["attacks"] = node_config[
                    "attacks"
                ]
                participant_config["adversarial_args"][
                    "poisoned_sample_percent"
                ] = node_config["poisoned_sample_percent"]
                participant_config["adversarial_args"]["poisoned_ratio"] = node_config[
                    "poisoned_ratio"
                ]
                participant_config["defense_args"]["with_reputation"] = data[
                    "with_reputation"
                ]
                participant_config["defense_args"]["is_dynamic_topology"] = data[
                    "is_dynamic_topology"
                ]
                participant_config["defense_args"]["is_dynamic_aggregation"] = data[
                    "is_dynamic_aggregation"
                ]
                # participant_config["defense_args"]["target_aggregation"] = data[
                #     "target_aggregation"
                # ]

                # Zi Ye
                # Begin
                participant_config["defense_args"]["dynamic_aggregation_mode"] = data[
                    "dynamic_aggregation_mode"
                ]
                # End
                

                participant_config["mobility_args"]["random_geo"] = data["random_geo"]
                participant_config["mobility_args"]["latitude"] = data["latitude"]
                participant_config["mobility_args"]["longitude"] = data["longitude"]
                # Get mobility config for each node (after applying the percentage from the frontend)
                participant_config["mobility_args"]["mobility"] = node_config["mobility"]
                participant_config["mobility_args"]["mobility_type"] = data["mobility_type"]
                participant_config["mobility_args"]["radius_federation"] = data[
                    "radius_federation"
                ]
                participant_config["mobility_args"]["scheme_mobility"] = data[
                    "scheme_mobility"
                ]
                participant_config["mobility_args"]["round_frequency"] = data["round_frequency"]
                

                with open(participant_file, "w") as f:
                    json.dump(participant_config, f, sort_keys=False, indent=2)

            # Create a argparse object
            import argparse
            import subprocess

            args_controller = argparse.Namespace(**args_controller)
            controller = Controller(
                args_controller
            )  # Generate an instance of controller in this new process
            try:
                if mobility_status:
                    additional_participants = data["additional_participants"]  # List of additional participants with dict("round": int). Example: [{"round": 1}, {"round": 2}]
                    schema_additional_participants = data["schema_additional_participants"]
                    print("additional_participants", additional_participants)
                    print("schema_additional_participants", schema_additional_participants)
                    controller.load_configurations_and_start_nodes(additional_participants, schema_additional_participants)
                else:
                    controller.load_configurations_and_start_nodes()
            except subprocess.CalledProcessError as e:
                print("Error docker-compose up:", e)
                return redirect(url_for("fedstellar_scenario_deployment"))
            # Generate/Update the scenario in the database
            scenario_update_record(
                scenario_name=controller.scenario_name,
                start_time=controller.start_date_scenario,
                end_time="",
                status="running",
                title=data["scenario_title"],
                description=data["scenario_description"],
                network_subnet=data["network_subnet"],
                model=data["model"],
                dataset=data["dataset"],
                rounds=data["rounds"],
                role=session["role"],
            )
            return redirect(url_for("fedstellar_scenario"))
        else:
            return abort(401)
    else:
        return abort(401)


if __name__ == "__main__":
    # Parse args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the frontend on."
    )
    args = parser.parse_args()
    print(f"Starting frontend on port {args.port}")
    # app.run(debug=True, host="0.0.0.0", port=int(args.port))
    # Get env variables
    socketio.run(app, debug=True, host="0.0.0.0", port=int(args.port))
