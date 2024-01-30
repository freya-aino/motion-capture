import torch as T


def anno_wrapper(func):
    def wrap(*args, **kwargs):

        assert "service_name" in kwargs, "service_name has to be given as kwarg"
        assert "video_name" in kwargs, "video_name has to be given as kwarg"

        local_anno_path = os.path.join(ANNOTATION_PATH, kwargs["service_name"], kwargs["video_name"])
        if not os.path.exists(local_anno_path):
            print("creating", local_anno_path)
            os.mkdir(local_anno_path)

        try: 
            print("----------------  ANNOTATE", kwargs["video_name"], "WITH", kwargs["service_name"], " ----------------")
            func(*args, **kwargs)
            print("---------------------------------  DONE ---------------------------------")
        except:
            traceback.print_exc()
            time.sleep(1)
            print("deleting: ", local_anno_path)
            if os.path.exists(local_anno_path):
                shutil.rmtree(local_anno_path)
            exit()
            
    return wrap


# ----------------------- OpenPose ----------------------- #

@anno_wrapper
def openpose_annotate(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    proj_path = Path(__file__).absolute().parent.joinpath("external_repo_cache", "openpose")


    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):

        cam_id = cam_name.split(".")[0]

        print("annotating video and saving results...")

        cmd = [
            "cd " + str(proj_path) + " &",
            ".\\build\\x64\\Release\\OpenPoseDemo.exe",
            "--video " + os.path.join(VIDEO_PATH, video_name, cam_name),
            "--write_json " + os.path.join(local_anno_path, cam_id),
            "--face",
            "--hand",
            "--display 0",
            "--render_pose 0",
            "--number_people_max 1"
        ]
        os.system(" ".join(cmd))

        print("format json ...")
        out = []
        for ji in os.listdir(os.path.join(local_anno_path, cam_id)):
            
            frame = int(ji.split("_")[1])

            # try:
            with open(os.path.join(local_anno_path, cam_id, ji), "r+") as ff:
                J = json.load(ff)
            # except Exception as e:
            #     if e.__class__ != json.decoder.JSONDecodeError:
            #         traceback.print_exc()
            #         exit()
            #     else:
            #         print("frame", frame, "data unavailable")
            #         out.append({
            #             "Frame": frame,
            #             "Keypoints_pose": None,
            #             "Keypoints_face": None,
            #             "Keypoints_left_hand": None,
            #             "Keypoints_hand_right": None
            #         })
            #         continue
            

            if "people" in J and len(J["people"]) > 0:
                pose_keypoints_2d = np.array(J["people"][0]["pose_keypoints_2d"]).reshape(25, 3) if "pose_keypoints_2d" in J["people"][0] else None
                face_keypoints_2d = np.array(J["people"][0]["face_keypoints_2d"]).reshape(70, 3) if "face_keypoints_2d" in J["people"][0] else None
                hand_left_keypoints_2d = np.array(J["people"][0]["hand_left_keypoints_2d"]).reshape(21, 3) if "hand_left_keypoints_2d" in J["people"][0] else None
                hand_right_keypoints_2d = np.array(J["people"][0]["hand_right_keypoints_2d"]).reshape(21, 3) if "hand_right_keypoints_2d" in J["people"][0] else None
                
                pose_keypoints_2d, pose_scores_2d = (pose_keypoints_2d[:, :2].flatten(), pose_keypoints_2d[:, 2].flatten()) if type(pose_keypoints_2d) == np.ndarray else (None, None)
                face_keypoints_2d, face_scores_2d = (face_keypoints_2d[:, :2].flatten(), face_keypoints_2d[:, 2].flatten()) if type(face_keypoints_2d) == np.ndarray else (None, None)
                hand_left_keypoints_2d, hand_left_scores_2d = (hand_left_keypoints_2d[:, :2].flatten(), hand_left_keypoints_2d[:, 2].flatten()) if type(hand_left_keypoints_2d) == np.ndarray else (None, None)
                hand_right_keypoints_2d, hand_right_scores_2d = (hand_right_keypoints_2d[:, :2].flatten(), hand_right_keypoints_2d[:, 2].flatten()) if type(hand_right_keypoints_2d) == np.ndarray else (None, None)

            else:
                pose_keypoints_2d, pose_scores_2d = None, None
                face_keypoints_2d, face_scores_2d = None, None
                hand_left_keypoints_2d, hand_left_scores_2d = None, None
                hand_right_keypoints_2d, hand_right_scores_2d = None, None

            out.append({
                "Frame": frame,
                "Keypoints_pose": pose_keypoints_2d,
                "Scores_pose": pose_scores_2d,
                "Keypoints_face": face_keypoints_2d,
                "Scores_face": face_scores_2d,
                "Keypoints_left_hand": hand_left_keypoints_2d,
                "Scores_left_hand": hand_left_scores_2d,
                "Keypoints_right_hand": hand_right_keypoints_2d,
                "Scores_right_hand": hand_right_scores_2d
            })
        out = pd.DataFrame.from_records(out)
        fit_table = pd.DataFrame.from_records([{"Frame": i} for i in range(max(out["Frame"]))])
        out = pd.merge(fit_table, out, how = "left", on = "Frame")

        print("saving as ", os.path.join(local_anno_path, cam_id + ".json"), "...")
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))

        print("cleanup ...")
        shutil.rmtree(os.path.join(local_anno_path, cam_id))

        print("successfully annotated:", video_name + "\\" + cam_name)


def prepare_op(x):

    availability = T.tensor(reduce(lambda a, b: a & b, [~x[k].isna() for k in x.keys()]), dtype = T.bool)
    
    keypoints_pose = T.stack([T.tensor(list(x["Keypoints_pose"].iloc[i])).reshape(25, 2) if availability[i] else T.nan * T.ones(25, 2) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_face = T.stack([T.tensor(list(x["Keypoints_face"].iloc[i])).reshape(70, 2) if availability[i] else T.nan * T.ones(70, 2) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_left_hand = T.stack([T.tensor(list(x["Keypoints_left_hand"].iloc[i])).reshape(21, 2) if availability[i] else T.nan * T.ones(21, 2) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_right_hand = T.stack([T.tensor(list(x["Keypoints_right_hand"].iloc[i])).reshape(21, 2) if availability[i] else T.nan * T.ones(21, 2) for i in range(len(x))]).to(dtype = T.float32)

    scores_pose = T.stack([T.tensor(list(x["Scores_pose"].iloc[i])) if availability[i] else T.nan * T.ones(25) for i in range(len(x))]).to(dtype = T.float32)
    scores_face = T.stack([T.tensor(list(x["Scores_face"].iloc[i])) if availability[i] else T.nan * T.ones(70) for i in range(len(x))]).to(dtype = T.float32)
    scores_left_hand = T.stack([T.tensor(list(x["Scores_left_hand"].iloc[i])) if availability[i] else T.nan * T.ones(21) for i in range(len(x))]).to(dtype = T.float32)
    scores_right_hand = T.stack([T.tensor(list(x["Scores_right_hand"].iloc[i])) if availability[i] else T.nan * T.ones(21) for i in range(len(x))]).to(dtype = T.float32)

    visibilities_pose = ~(keypoints_pose == 0.).all(-1)
    visibilities_face = ~(keypoints_face == 0.).all(-1)
    visibilities_left_hand = ~(keypoints_left_hand == 0.).all(-1)
    visibilities_right_hand = ~(keypoints_right_hand == 0.).all(-1)

    keypoints_pose[~visibilities_pose] = T.nan
    keypoints_face[~visibilities_face] = T.nan
    keypoints_left_hand[~visibilities_left_hand] = T.nan
    keypoints_right_hand[~visibilities_right_hand] = T.nan

    print("TODO: check if face keypoints lsat 5 elements are usefull")

    return {
        "availability": availability,
        "keypoints_pose": keypoints_pose,
        "keypoints_face": keypoints_face,
        "keypoints_left_hand": keypoints_left_hand,
        "keypoints_right_hand": keypoints_right_hand,
        "scores_pose": scores_pose,
        "scores_face": scores_face,
        "scores_left_hand": scores_left_hand,
        "scores_right_hand": scores_right_hand,
        "visibilities_pose": visibilities_pose,
        "visibilities_face": visibilities_face,
        "visibilities_left_hand": visibilities_left_hand,
        "visibilities_right_hand": visibilities_right_hand,
    }



# ----------------------- Mediapipe ----------------------- #



@anno_wrapper
def mediapipe_annotation(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)

    from mediapipe.python.solutions.hands import Hands

    hand_tracker = Hands()

    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):
        cam_id = cam_name.split(".")[0]

        video_frames = np.uint8(load_all_frames(os.path.join(VIDEO_PATH, video_name, cam_name)))

        out = []

        print("annotating     [", end = "")
        for frame in video_frames:

            print("|", end = "", flush = True)

            # frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hand_tracker.process(frame)

            oo = {
                "Score_right_hand": None,
                "Score_left_hand": None,
                "Keypoints_right_hand": None,
                "Keypoints_left_hand": None,
                "Keypoints_left_hand_local": None,
                "Keypoints_right_hand_local": None,
            }

            if result.multi_handedness != None:
                for i in range(len(result.multi_handedness)):
                    if result.multi_handedness[i].classification[0].label == "Right":
                        oo["Score_left_hand"] = result.multi_handedness[i].classification[0].score
                        oo["Keypoints_left_hand"] = list(np.array([[l.x * CAMERA_OUT_WIDTH, l.y * CAMERA_OUT_HEIGHT, l.z] for l in result.multi_hand_landmarks[i].landmark]).flatten())
                        oo["Keypoints_left_hand_local"] = list(np.array([[l.x, l.y, l.z] for l in result.multi_hand_world_landmarks[i].landmark]).flatten())
                    elif result.multi_handedness[i].classification[0].label == "Left":
                        oo["Score_right_hand"] = result.multi_handedness[i].classification[0].score
                        oo["Keypoints_right_hand"] = list(np.array([[l.x * CAMERA_OUT_WIDTH, l.y * CAMERA_OUT_HEIGHT, l.z] for l in result.multi_hand_landmarks[i].landmark]).flatten())
                        oo["Keypoints_right_hand_local"] = list(np.array([[l.x, l.y, l.z] for l in result.multi_hand_world_landmarks[i].landmark]).flatten())
            
            out.append(oo)

        print("\nsaving outputs to:", os.path.join(local_anno_path, cam_id + ".json"), "...")
        out = pd.DataFrame.from_records(out)
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))


def prepare_mediapipe(x):

    right_mask_list = [~x["Keypoints_right_hand"].isna(), ~x["Score_right_hand"].isna()]
    right_availability = T.tensor(reduce(lambda a, b: a & b, right_mask_list), dtype = T.bool)
    
    left_mask_list = [~x["Keypoints_left_hand"].isna(), ~x["Score_left_hand"].isna()]
    left_availability = T.tensor(reduce(lambda a, b: a & b, left_mask_list), dtype = T.bool)

    
    scores_right_hand = T.tensor([x["Score_right_hand"].iloc[i] if right_availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32)
    scores_left_hand = T.tensor([x["Score_left_hand"].iloc[i] if left_availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32)

    keypoints_right_hand = T.stack([T.tensor(list(x["Keypoints_right_hand"].iloc[i])).reshape(21, 3) if right_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_left_hand = T.stack([T.tensor(list(x["Keypoints_left_hand"].iloc[i])).reshape(21, 3) if left_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)

    keypoints_right_hand_local = T.stack([T.tensor(list(x["Keypoints_right_hand_local"].iloc[i])).reshape(21, 3) if right_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)
    keypoints_left_hand_local = T.stack([T.tensor(list(x["Keypoints_left_hand_local"].iloc[i])).reshape(21, 3) if left_availability[i] else T.nan * T.ones(21, 3) for i in range(len(x))]).to(dtype = T.float32)

    # keypoints_right_hand[:, 0] = keypoints_right_hand[:, 0] * 1920
    # keypoints_right_hand[:, 1] = keypoints_right_hand[:, 1] * 1080

    return {
        "availability_right_hand": right_availability,
        "availability_left_hand": left_availability,
        "scores_right_hand": scores_right_hand,
        "scores_left_hand": scores_left_hand,
        "keypoints_right_hand": keypoints_right_hand,
        "keypoints_left_hand": keypoints_left_hand,
        "keypoints_right_hand_local_space": keypoints_right_hand_local,
        "keypoints_left_hand_local_space": keypoints_left_hand_local,
    }



def mp_kpts_prep(
    mp_image_space,
    mp_local_space,
    icp_max_iters = 10,
    icp_samples = 30):

    mp_gn = ms_norm(mp_image_space, dim = 0)
    mp_ln = ms_norm(mp_local_space, dim = 0) 
    mp_valid_mask = ~mp_gn.isnan().any(-1) & ~mp_ln.isnan().any(-1)

    if mp_valid_mask.any():
        mp_ln[mp_valid_mask] = T.tensor(stochasticICP_search(mp_ln[mp_valid_mask].numpy().T, mp_gn[mp_valid_mask].numpy().T, icp_max_iters, icp_samples), dtype=T.float32)
    # mp_ln = T.tensor(stochasticICP_search(mp_ln.numpy().T, mp_gn.numpy().T, icp_max_iters, icp_samples), dtype=T.float32)

    mp_ln = ms_norm(mp_ln, dim = 0)
    mp_ln = mp_ln - mp_ln[0]  # 0 = wrist

    return mp_ln




# ----------------------- OpenSeeFace ----------------------- #


@anno_wrapper
def open_see_face_annotate(video_name, service_name):
    local_anno_path = os.path.join(ANNOTATION_PATH, service_name, video_name)
    
    proj_path = Path(__file__).absolute().parent.joinpath("external_repo_cache", "OpenSeeFace_altered")
    sys.path.append(str(proj_path))
    
    from external_repo_cache.OpenSeeFace_altered.tracker import Tracker as open_see_face_tracker
    
    model_dir = os.path.join(proj_path, "models")
    
    face_tracker = open_see_face_tracker(
        CAMERA_OUT_WIDTH, 
        CAMERA_OUT_HEIGHT, 
        model_type = 4,
        model_dir = model_dir, 
        silent = True)
    
    for cam_name in os.listdir(os.path.join(VIDEO_PATH, video_name)):
        
        print("annotating video", os.path.join(VIDEO_PATH, video_name, cam_name))
        
        cam_id = cam_name.split(".")[0]
        v_path = os.path.join(VIDEO_PATH, video_name, cam_name)
        frames = load_all_frames(v_path)
        out = []
        
        for i in range(frames.shape[0]):
            
            oo = {
                "Frame": i,
                "Success": None,
                "Conf": None,
                "Keypoints": None,
                "Quaternion": None,
                "Euler": None,
                "Translation": None,
                "Bbox": None,
                "Eye_r": None,
                "Eye_r_conf": None,
                "Eye_l": None,
                "Eye_l_conf": None
            }
            
            face_result = face_tracker.predict(frames[i])
            
            if len(face_result) > 0:
                
                face_result = face_result[0]
                if face_result != None:
                    
                    success = face_result.success
                    lms = face_result.lms.flatten()
                    lms_3d = face_result.pts_3d.flatten()
                    quaternion = face_result.quaternion
                    euler = face_result.euler
                    translation = face_result.translation
                    bbox = face_result.bbox
                    conf = face_result.conf
                    
                    _, eye_r_y, eye_r_x, eye_r_conf = face_result.eye_state[0]
                    _, eye_l_y, eye_l_x, eye_l_conf = face_result.eye_state[1]

                    out.append({
                        "Frame": i,
                        "Success": success,
                        "Conf": conf,
                        "Keypoints": lms,
                        "Keypoints_3D": lms_3d,
                        "Quaternion": quaternion,
                        "Euler": euler,
                        "Translation": translation,
                        "Bbox": bbox,
                        "Eye_r": [eye_r_x, eye_r_y],
                        "Eye_r_conf": eye_r_conf,
                        "Eye_l": [eye_l_x, eye_l_y],
                        "Eye_l_conf": eye_l_conf
                    })
                else:
                    out.append(oo)

        out = pd.DataFrame.from_records(out)

        print("saving to", os.path.join(local_anno_path, cam_id + ".json"))
        out.to_json(os.path.join(local_anno_path, cam_id + ".json"))


def prepare_osf(x, face_anchor_offset = 0.6):

    x.loc[x["Success"].isna(), "Success"] = 0.
    x["Success"] = x["Success"].astype(bool)
    
    availability = T.tensor(reduce(lambda a, b: a & b, [x["Success"], *[~x[k].isna() for k in x.keys()]]), dtype = T.bool)

    kpts = T.stack([T.tensor(list(x["Keypoints"].iloc[i])).reshape(68, 3) if availability[i] else T.nan * T.ones(68, 3) for i in range(len(x))]).to(dtype = T.float32)
    kpts_2d = kpts[..., :2] # .flip(-1)
    kpts_confidences = kpts[..., 2]


    kpts_3d_image_space = T.stack([T.tensor(list(x["Keypoints_3D"].iloc[i])).reshape(70, 3) if availability[i] else T.nan * T.ones(70, 3) for i in range(len(x))]).to(dtype = T.float32)


    static_face_ancor = kpts_3d_image_space[:, 27:36, :].nanmean(1, keepdim = True) - T.tensor([0, 0, face_anchor_offset])
    kpts_3d_image_space = T.cat([kpts_3d_image_space, static_face_ancor], dim = 1)
    kpts_3d_local_space = deepcopy(kpts_3d_image_space)

    # T.tensor(info[x[c]["Frame"] == jj].iloc[0])) if availability[jj] else T.nan * T.ones(27) for jj in range(num_frames)]).to(dtype = T.float32)
    # bbox = T.stack([T.tensor(list(x["Bbox"].iloc[i])) if availability[i] else T.nan * T.ones(4) for i in range(len(x))]).to(dtype = T.float32)
    translation = T.stack([T.tensor(list(x["Translation"].iloc[i])) if availability[i] else T.nan * T.ones(3) for i in range(len(x))]).to(dtype = T.float32)
    # euler = T.stack([T.tensor(list(x["Euler"].iloc[i])) if availability[i] else T.nan * T.ones(3) for i in range(len(x))]).to(dtype = T.float32)
    quaternion = T.stack([T.tensor(list(x["Quaternion"].iloc[i])) if availability[i] else T.nan * T.ones(4) for i in range(len(x))]).to(dtype = T.float32)


    from scipy.spatial.transform import Rotation as sci_rot

    camera_rot = np.array([[CAMERA_OUT_WIDTH, 0, CAMERA_OUT_WIDTH/2], [0, CAMERA_OUT_WIDTH, CAMERA_OUT_HEIGHT/2], [0, 0, 1]], dtype = np.float32)

    for i in range(kpts_3d_image_space.shape[0]):
        if availability[i]:
            pt_3d = kpts_3d_image_space[i] # , [*range(0, 66), 68, 69]]
            
            rot_mat_q = sci_rot.from_quat(quaternion[i].numpy()).as_matrix()
            pt_3d[:, 0] = -pt_3d[:, 0]
            pt_3d = pt_3d.numpy().dot(rot_mat_q)
            pt_3d = pt_3d + translation[i].numpy()
            pt_3d = pt_3d.dot(camera_rot.transpose())
            pt_3d = T.tensor(pt_3d, dtype = T.float32)

            depth = pt_3d[:, 2].unsqueeze(-1)
            depth[depth == 0] = 0.0001
            pt_3d[:, :2] = pt_3d[:, :2] / depth
            # pt_3d[:, :2] = kpts_2d[i]

            kpts_3d_image_space[i, :] = pt_3d[:]


    kpts_3d_image_space[..., :2] = kpts_3d_image_space[..., :2].flip(-1)
    kpts_2d[..., :2] = kpts_2d[..., :2].flip(-1)

    return {
        "availability": availability,
        "confidences": T.tensor([x["Conf"].iloc[i] if availability[i] else T.nan for i in range(len(x))]).to(dtype = T.float32),
        "keypoints_2d": kpts_2d,
        "keypoints_confidences": kpts_confidences,
        "keypoints_3d_image_space": kpts_3d_image_space,
        "keypoints_3d_local_space": kpts_3d_local_space,
        # "info": T.cat([bbox, translation, euler], dim = -1),
    }
    
def osf_kpts_prep(
    osf_image_space,
    osf_local_space,
    icp_max_iters = 10,
    icp_samples = 30):

    osf_gn = ms_norm(osf_image_space, dim = 0)
    osf_ln = ms_norm(osf_local_space, dim = 0) 
    osf_valid_mask = ~osf_gn.isnan().any(-1) & ~osf_ln.isnan().any(-1)

    if osf_valid_mask.any():
        osf_ln[osf_valid_mask] = T.tensor(stochasticICP_search(osf_ln[osf_valid_mask].numpy().T, osf_gn[osf_valid_mask].numpy().T, icp_max_iters, icp_samples), dtype=T.float32)
    # osf_ln = T.tensor(stochasticICP_search(osf_ln.numpy().T, osf_gn.numpy().T, icp_max_iters, icp_samples), dtype=T.float32)

    osf_ln = ms_norm(osf_ln, dim = 0)
    osf_ln = osf_ln - osf_ln[78] # 79 = face anchor

    return osf_ln