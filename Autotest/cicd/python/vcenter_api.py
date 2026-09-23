#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vcenter_api.py - Interacción con vCenter REST API (sin pyvmomi)

Compatible con Python 3.6+
Usa la API REST de vCenter 6.5+ directamente con requests

Uso:
    python3.6 vcenter_api.py <config_path> <action> [args...]
    
Acciones:
    upload_iso <local_iso_path>      - Subir ISO al datastore
    configure_cdrom                  - Configurar CD-ROM de la VM con el ISO
    power_on                         - Encender la VM
    power_off                        - Apagar la VM
    get_vm_status                    - Obtener estado de la VM
    revert_snapshot [snapshot_name]  - Revertir al snapshot (usa config si no se pasa arg)
    wait_powered_off                 - Esperar a que la VM esté apagada
    wait_powered_on                  - Esperar a que la VM esté encendida
"""

from __future__ import print_function
import sys
import os
import yaml
import requests
import json
import time
import urllib3

# Desactivar warnings SSL para entornos con certificados auto-firmados
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DEFAULT_REQUEST_TIMEOUT = 60      # segundos por petición REST/SOAP
DEFAULT_UPLOAD_TIMEOUT = 7200     # segundos sin actividad de red durante la subida del ISO


def _positive_int(value, default):
    """Convertir un valor de configuración a entero positivo, o devolver default"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class _HttpClient(object):
    """Envoltorio de requests que aplica un timeout por defecto a cada llamada.

    requests no tiene timeout por defecto: sin él, una petición a vCenter que
    no responde deja el pipeline colgado indefinidamente. Las llamadas que
    pasan 'timeout' de forma explícita lo conservan.
    """

    def __init__(self, timeout):
        self.timeout = timeout

    def request(self, method, url, **kwargs):
        kwargs.setdefault('timeout', self.timeout)
        return requests.request(method, url, **kwargs)

    def get(self, url, **kwargs):
        return self.request('GET', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)

    def put(self, url, **kwargs):
        return self.request('PUT', url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request('PATCH', url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request('DELETE', url, **kwargs)


class VCenterRESTClient(object):
    """Cliente para vCenter REST API (compatible con Python 3.6)"""
    
    def __init__(self, config):
        self.config = config.get('vcenter', {})
        self.base_url = self.config.get('url', '').rstrip('/')
        self.username = os.environ.get('VCENTER_USER', self.config.get('username', ''))
        self.password = os.environ.get('VCENTER_PASSWORD', self.config.get('password', ''))
        self.session_id = None
        self.verify_ssl = False  # Cambiar a True si tienes certificados válidos
        self.request_timeout = _positive_int(
            self.config.get('request_timeout_seconds'), DEFAULT_REQUEST_TIMEOUT)
        self.upload_timeout = _positive_int(
            self.config.get('upload_timeout_seconds'), DEFAULT_UPLOAD_TIMEOUT)
        self.http = _HttpClient(self.request_timeout)
        
    def _get_headers(self):
        """Obtener headers para peticiones autenticadas"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if self.session_id:
            headers['vmware-api-session-id'] = self.session_id
        return headers
    
    def connect(self):
        """Autenticar y obtener session ID"""
        url = '{}/api/session'.format(self.base_url)
        
        try:
            response = self.http.post(
                url,
                auth=(self.username, self.password),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            # La respuesta es el session ID como string JSON
            self.session_id = response.json()
            print('[OK] Conectado a vCenter: {}'.format(self.base_url))
            return True
            
        except requests.exceptions.RequestException as e:
            # Intentar con endpoint alternativo (versiones antiguas)
            return self._connect_legacy()
    
    def _connect_legacy(self):
        """Autenticación para vCenter 6.5/6.7 (endpoint antiguo)"""
        url = '{}/rest/com/vmware/cis/session'.format(self.base_url)
        
        try:
            response = self.http.post(
                url,
                auth=(self.username, self.password),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            
            data = response.json()
            self.session_id = data.get('value', '')
            print('[OK] Conectado a vCenter (legacy): {}'.format(self.base_url))
            return True
            
        except requests.exceptions.RequestException as e:
            print('[ERROR] No se pudo conectar a vCenter: {}'.format(str(e)))
            raise
    
    def disconnect(self):
        """Cerrar sesión"""
        if not self.session_id:
            return
            
        try:
            url = '{}/api/session'.format(self.base_url)
            self.http.delete(url, headers=self._get_headers(), verify=self.verify_ssl)
            print('[OK] Sesión cerrada')
        except Exception:
            pass  # Ignorar errores al cerrar
        finally:
            self.session_id = None
    
    def get_vm(self, vm_name):
        """Obtener VM por nombre"""
        url = '{}/api/vcenter/vm'.format(self.base_url)
        params = {'names': vm_name}
        
        response = self.http.get(
            url,
            headers=self._get_headers(),
            params=params,
            verify=self.verify_ssl
        )
        
        if response.status_code == 404:
            # Intentar endpoint legacy
            return self._get_vm_legacy(vm_name)
        
        response.raise_for_status()
        vms = response.json()
        
        if not vms:
            raise Exception('VM no encontrada: {}'.format(vm_name))
        
        return vms[0]
    
    def _get_vm_legacy(self, vm_name):
        """Obtener VM (endpoint legacy)"""
        url = '{}/rest/vcenter/vm'.format(self.base_url)
        params = {'filter.names': vm_name}
        
        response = self.http.get(
            url,
            headers=self._get_headers(),
            params=params,
            verify=self.verify_ssl
        )
        response.raise_for_status()
        
        data = response.json()
        vms = data.get('value', [])
        
        if not vms:
            raise Exception('VM no encontrada: {}'.format(vm_name))
        
        return vms[0]
    
    def get_vm_power_state(self, vm_id):
        """Obtener estado de power de la VM"""
        url = '{}/api/vcenter/vm/{}/power'.format(self.base_url, vm_id)
        
        response = self.http.get(
            url,
            headers=self._get_headers(),
            verify=self.verify_ssl
        )
        response.raise_for_status()
        
        return response.json().get('state', 'UNKNOWN')
    
    def power_on_vm(self, vm_name=None):
        """Encender la VM"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        # Verificar estado actual
        try:
            state = self.get_vm_power_state(vm_id)
            if state == 'POWERED_ON':
                print('[*] VM ya encendida: {}'.format(vm_name))
                return True
        except Exception:
            pass  # Continuar con power on
        
        url = '{}/api/vcenter/vm/{}/power?action=start'.format(self.base_url, vm_id)
        
        try:
            response = self.http.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        except Exception:
            # Intentar endpoint legacy
            url = '{}/rest/vcenter/vm/{}/power/start'.format(self.base_url, vm_id)
            response = self.http.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        
        print('[OK] VM encendida: {}'.format(vm_name))
        return True
    
    def power_off_vm(self, vm_name=None):
        """Apagar la VM"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        url = '{}/api/vcenter/vm/{}/power?action=stop'.format(self.base_url, vm_id)
        
        try:
            response = self.http.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        except Exception:
            # Intentar endpoint legacy
            url = '{}/rest/vcenter/vm/{}/power/stop'.format(self.base_url, vm_id)
            response = self.http.post(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
        
        print('[OK] VM apagada: {}'.format(vm_name))
        return True
    
    def upload_iso_to_datastore(self, local_iso_path, remote_filename=None):
        """
        Subir ISO al datastore vía HTTP PUT
        
        Usa el endpoint de ficheros del datastore:
        PUT https://vcenter/folder/<path>?dcPath=<datacenter>&dsName=<datastore>
        """
        datastore = self.config.get('datastore')
        datacenter = self.config.get('datacenter')
        iso_folder = self.config.get('iso_path', '/ISO').lstrip('/')
        
        if remote_filename is None:
            remote_filename = os.path.basename(local_iso_path)
        
        remote_path = '{}/{}'.format(iso_folder, remote_filename)
        
        # URL para upload via HTTPS
        url = '{}/folder/{}?dcPath={}&dsName={}'.format(
            self.base_url,
            remote_path,
            datacenter,
            datastore
        )
        
        file_size = os.path.getsize(local_iso_path)
        print('[*] Subiendo ISO: {} ({:.2f} GB)'.format(
            local_iso_path, 
            file_size / (1024**3)
        ))
        print('[*] Destino: [{}] {}'.format(datastore, remote_path))
        
        # Subir con streaming para ficheros grandes
        with open(local_iso_path, 'rb') as f:
            # (conexión, lectura): la lectura cubre también el tiempo que
            # vCenter tarda en responder cuando termina de recibir el fichero
            response = self.http.put(
                url,
                data=f,
                auth=(self.username, self.password),
                verify=self.verify_ssl,
                timeout=(self.request_timeout, self.upload_timeout),
                headers={
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(file_size)
                }
            )
        
        if response.status_code in [200, 201, 204]:
            print('[OK] ISO subido: [{}] {}'.format(datastore, remote_path))
            return '[{}] {}'.format(datastore, remote_path)
        else:
            raise Exception('Error subiendo ISO: {} - {}'.format(
                response.status_code, 
                response.text
            ))
    
    def get_vm_hardware(self, vm_id):
        """Obtener configuración de hardware de la VM"""
        url = '{}/api/vcenter/vm/{}/hardware'.format(self.base_url, vm_id)
        
        response = self.http.get(
            url,
            headers=self._get_headers(),
            verify=self.verify_ssl
        )
        response.raise_for_status()
        
        return response.json()
    
    def get_cdrom_devices(self, vm_id):
        """Obtener lista de dispositivos CD-ROM de la VM"""
        url = '{}/api/vcenter/vm/{}/hardware/cdrom'.format(self.base_url, vm_id)
        
        try:
            response = self.http.get(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            # Endpoint legacy
            url = '{}/rest/vcenter/vm/{}/hardware/cdrom'.format(self.base_url, vm_id)
            response = self.http.get(
                url,
                headers=self._get_headers(),
                verify=self.verify_ssl
            )
            response.raise_for_status()
            return response.json().get('value', [])
    
    def configure_cdrom(self, vm_name=None, iso_path=None):
        """
        Configurar CD-ROM de la VM para usar el ISO
        
        iso_path: Ruta en formato [datastore] path/to/file.iso
        """
        vm_name = vm_name or self.config.get('vm_name')
        
        if iso_path is None:
            datastore = self.config.get('datastore')
            iso_folder = self.config.get('iso_path', '/ISO').lstrip('/')
            iso_path = '[{}] {}/{}.iso'.format(datastore, iso_folder, vm_name)
        
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        # Obtener CD-ROMs de la VM
        cdroms = self.get_cdrom_devices(vm_id)
        
        if not cdroms:
            raise Exception('No se encontró dispositivo CD-ROM en la VM')
        
        # Usar el primer CD-ROM
        cdrom = cdroms[0]
        cdrom_id = cdrom.get('cdrom', cdrom.get('key'))
        
        print('[*] Configurando CD-ROM (id: {})'.format(cdrom_id))
        print('[*] ISO objetivo: {}'.format(iso_path))
        
        # Esperar un momento para que el SO libere el CD-ROM (debe desmontarse antes desde SSH)
        import time
        print('[*] Esperando 2 segundos para que el SO libere el CD-ROM...')
        time.sleep(2)
        
        # PASO 1: Desconectar el CD-ROM (intentar, pero continuar si falla)
        # NOTA: El SO invitado debe haber ejecutado 'eject' antes de llegar aquí
        # para liberar el lock del kernel. Si el SO sigue bloqueando el CD-ROM,
        # vCenter mostrará un diálogo de confirmación que puede colgar el pipeline.
        # El timeout de 30s evita que la llamada se quede bloqueada indefinidamente.
        print('[*] Desconectando CD-ROM...')
        disconnected = False
        try:
            disconnect_url = '{}/rest/vcenter/vm/{}/hardware/cdrom/{}/disconnect'.format(
                self.base_url, vm_id, cdrom_id
            )
            disconnect_response = self.http.post(
                disconnect_url,
                headers=self._get_headers(),
                verify=self.verify_ssl,
                timeout=30
            )
            if disconnect_response.status_code in [200, 204]:
                print('[OK] CD-ROM desconectado')
                disconnected = True
                time.sleep(2)  # Esperar a que se aplique la desconexión
            else:
                print('[WARN] Respuesta inesperada al desconectar: {}'.format(disconnect_response.status_code))
        except requests.exceptions.Timeout:
            print('[WARN] Timeout al desconectar CD-ROM (30s). El SO invitado puede seguir bloqueando el dispositivo.')
            print('[*] Asegurese de ejecutar "eject /dev/sr0" en el SO invitado antes de llamar a esta funcion.')
            print('[*] Intentando continuar con el cambio de ISO...')
        except Exception as e:
            print('[WARN] No se pudo desconectar CD-ROM: {}'.format(str(e)))
            print('[*] Intentando continuar con el cambio de ISO...')
        
        # PASO 2: Configurar el backing (cambiar el ISO)
        print('[*] Actualizando backing del CD-ROM...')
        url = '{}/rest/vcenter/vm/{}/hardware/cdrom/{}'.format(
            self.base_url, vm_id, cdrom_id
        )
        
        payload = {
            'spec': {
                'backing': {
                    'type': 'ISO_FILE',
                    'iso_file': iso_path
                },
                'start_connected': True,
                'allow_guest_control': True
            }
        }
        
        try:
            response = self.http.patch(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            print('[OK] Backing del CD-ROM actualizado')
        except requests.exceptions.Timeout:
            raise Exception('Timeout al actualizar backing del CD-ROM (30s). Verifique que el SO invitado ejecuto "eject" antes.')
        except requests.exceptions.HTTPError as e:
            print('[WARN] API REST falló, intentando con API moderna...')
            # Intentar con API moderna
            url = '{}/api/vcenter/vm/{}/hardware/cdrom/{}'.format(
                self.base_url, vm_id, cdrom_id
            )
            payload = {
                'backing': {
                    'type': 'ISO_FILE',
                    'iso_file': iso_path
                },
                'start_connected': True,
                'allow_guest_control': True
            }
            response = self.http.patch(
                url,
                headers=self._get_headers(),
                json=payload,
                verify=self.verify_ssl,
                timeout=30
            )
            response.raise_for_status()
            print('[OK] Backing del CD-ROM actualizado (API moderna)')
        
        time.sleep(2)  # Esperar a que se aplique el cambio
        
        # PASO 3: Conectar el CD-ROM explícitamente
        print('[*] Conectando CD-ROM...')
        max_retries = 3
        connected = False
        
        for attempt in range(1, max_retries + 1):
            try:
                connect_url = '{}/rest/vcenter/vm/{}/hardware/cdrom/{}/connect'.format(
                    self.base_url, vm_id, cdrom_id
                )
                connect_response = self.http.post(
                    connect_url,
                    headers=self._get_headers(),
                    verify=self.verify_ssl,
                    timeout=30
                )
                if connect_response.status_code in [200, 204]:
                    print('[OK] CD-ROM conectado (intento {}/{})'.format(attempt, max_retries))
                    connected = True
                    break
                elif connect_response.status_code == 400:
                    # Puede que ya esté conectado
                    print('[*] CD-ROM puede que ya esté conectado (intento {}/{})'.format(attempt, max_retries))
                    connected = True
                    break
                else:
                    print('[WARN] Respuesta inesperada al conectar: {} (intento {}/{})'.format(
                        connect_response.status_code, attempt, max_retries))
                    if attempt < max_retries:
                        time.sleep(2)
            except Exception as e:
                print('[WARN] Error conectando CD-ROM: {} (intento {}/{})'.format(str(e), attempt, max_retries))
                if attempt < max_retries:
                    time.sleep(2)
        
        if not connected:
            print('[ERROR] No se pudo conectar el CD-ROM después de {} intentos'.format(max_retries))
            raise Exception('Fallo al conectar CD-ROM después del cambio de ISO')
        
        print('[OK] CD-ROM configurado correctamente con ISO: {}'.format(iso_path))
        return True
    
    def list_snapshots(self, vm_id):
        """Obtener lista de snapshots de la VM (REST API, solo vCenter 8.0+)"""
        url = '{}/api/vcenter/vm/{}/snapshot'.format(self.base_url, vm_id)
        try:
            response = self.http.get(url, headers=self._get_headers(), verify=self.verify_ssl)
            response.raise_for_status()
            return response.json()
        except Exception:
            url = '{}/rest/vcenter/vm/{}/snapshot'.format(self.base_url, vm_id)
            response = self.http.get(url, headers=self._get_headers(), verify=self.verify_ssl)
            response.raise_for_status()
            return response.json().get('value', [])

    def revert_to_snapshot(self, vm_name=None, snapshot_name=None):
        """Revertir la VM al snapshot indicado"""
        vm_name = vm_name or self.config.get('vm_name')
        snapshot_name = snapshot_name or self.config.get('snapshot_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))

        try:
            snapshots = self.list_snapshots(vm_id)
        except Exception:
            # vCenter 6.x/7.x no tiene REST API para snapshots → usar SOAP
            print('[*] REST snapshot API no disponible, usando SOAP...')
            self._revert_snapshot_soap(vm_id, snapshot_name)
            print('[OK] Snapshot revertido via SOAP: {}'.format(snapshot_name))
            return True

        snapshot_id = None
        for s in snapshots:
            if s.get('name') == snapshot_name:
                snapshot_id = s.get('snapshot')
                break

        if not snapshot_id:
            raise Exception('Snapshot no encontrado: {}'.format(snapshot_name))

        url = '{}/api/vcenter/vm/{}/snapshot/{}?action=revert'.format(
            self.base_url, vm_id, snapshot_id)
        try:
            response = self.http.post(url, headers=self._get_headers(), verify=self.verify_ssl)
            if response.status_code not in [200, 204]:
                raise Exception('Status {}'.format(response.status_code))
        except Exception:
            url = '{}/rest/vcenter/vm/{}/snapshot/{}/revert'.format(
                self.base_url, vm_id, snapshot_id)
            response = self.http.post(url, headers=self._get_headers(), verify=self.verify_ssl)
            response.raise_for_status()

        print('[OK] Snapshot revertido: {}'.format(snapshot_name))
        return True

    def _soap_envelope(self, body_content):
        """Envolver contenido en un SOAP Envelope con prefijos explícitos vim25/soapenv"""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<soapenv:Envelope'
            ' xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"'
            ' xmlns:vim25="urn:vim25"'
            ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<soapenv:Body>{body}</soapenv:Body>'
            '</soapenv:Envelope>'
        ).format(body=body_content)

    def _parse_soap_fault(self, xml_text):
        """Extraer mensaje legible de una respuesta SOAP Fault"""
        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_text)
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag == 'faultstring':
                    return elem.text or 'sin detalle'
        except Exception:
            pass
        return (xml_text[:500] if xml_text else 'respuesta vacía')

    def _revert_snapshot_soap(self, vm_id, snapshot_name):
        """Revertir snapshot via SOAP API (vCenter sin REST snapshot API)"""
        import xml.etree.ElementTree as ET

        soap_url = '{}/sdk'.format(self.base_url)
        soap_headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': ''
        }

        # 1. Login SOAP con prefijos explícitos
        login_body = (
            '<vim25:Login>'
            '<vim25:_this type="SessionManager">SessionManager</vim25:_this>'
            '<vim25:userName>{user}</vim25:userName>'
            '<vim25:password>{password}</vim25:password>'
            '</vim25:Login>'
        ).format(user=self.username, password=self.password)

        resp = self.http.post(soap_url, data=self._soap_envelope(login_body),
                             headers=soap_headers, verify=self.verify_ssl)
        if not resp.ok:
            raise Exception('SOAP Login fallido ({}): {}'.format(
                resp.status_code, self._parse_soap_fault(resp.text)))
        cookies = resp.cookies

        # 2. Obtener árbol de snapshots via PropertyCollector
        # Usar RetrieveProperties (no Ex) para compatibilidad con API version <4.0
        prop_body = (
            '<vim25:RetrieveProperties>'
            '<vim25:_this type="PropertyCollector">propertyCollector</vim25:_this>'
            '<vim25:specSet>'
            '<vim25:propSet>'
            '<vim25:type>VirtualMachine</vim25:type>'
            '<vim25:pathSet>snapshot</vim25:pathSet>'
            '</vim25:propSet>'
            '<vim25:objectSet>'
            '<vim25:obj type="VirtualMachine">{vm_id}</vim25:obj>'
            '</vim25:objectSet>'
            '</vim25:specSet>'
            '</vim25:RetrieveProperties>'
        ).format(vm_id=vm_id)

        resp = self.http.post(soap_url, data=self._soap_envelope(prop_body),
                             headers=soap_headers, cookies=cookies, verify=self.verify_ssl)
        if not resp.ok:
            raise Exception('SOAP RetrieveProperties fallido ({}): {}'.format(
                resp.status_code, self._parse_soap_fault(resp.text)))

        root = ET.fromstring(resp.text)
        snapshot_moref = self._find_snapshot_in_xml(root, snapshot_name)

        if not snapshot_moref:
            raise Exception('Snapshot no encontrado via SOAP: {}'.format(snapshot_name))

        print('[*] Snapshot encontrado: {} -> {}'.format(snapshot_name, snapshot_moref))

        # 3. Revertir al snapshot
        revert_body = (
            '<vim25:RevertToSnapshot_Task>'
            '<vim25:_this type="VirtualMachineSnapshot">{snap}</vim25:_this>'
            '</vim25:RevertToSnapshot_Task>'
        ).format(snap=snapshot_moref)

        resp = self.http.post(soap_url, data=self._soap_envelope(revert_body),
                             headers=soap_headers, cookies=cookies, verify=self.verify_ssl)
        if not resp.ok:
            raise Exception('SOAP RevertToSnapshot fallido ({}): {}'.format(
                resp.status_code, self._parse_soap_fault(resp.text)))

        # 4. Esperar a que la tarea complete
        task_root = ET.fromstring(resp.text)
        task_moref = None
        for elem in task_root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'returnval':
                task_moref = elem.text
                break

        if task_moref:
            self._wait_for_soap_task(soap_url, soap_headers, cookies, task_moref)

    def _find_snapshot_in_xml(self, root, target_name):
        """Buscar recursivamente el moref de un snapshot por nombre en respuesta SOAP"""
        def search_entry(elem):
            name_text = None
            snap_moref = None
            child_lists = []
            for child in elem:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag == 'name':
                    name_text = child.text
                elif tag == 'snapshot':
                    snap_moref = child.text
                elif tag == 'childSnapshotList':
                    child_lists.append(child)
            if name_text == target_name and snap_moref:
                return snap_moref
            for cl in child_lists:
                result = search_entry(cl)
                if result:
                    return result
            return None

        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag == 'rootSnapshotList':
                result = search_entry(elem)
                if result:
                    return result
        return None

    def _wait_for_soap_task(self, soap_url, soap_headers, cookies, task_moref,
                            max_attempts=60, interval=5):
        """Esperar a que una tarea SOAP de vCenter complete"""
        import xml.etree.ElementTree as ET

        query_body = (
            '<vim25:RetrieveProperties>'
            '<vim25:_this type="PropertyCollector">propertyCollector</vim25:_this>'
            '<vim25:specSet>'
            '<vim25:propSet>'
            '<vim25:type>Task</vim25:type>'
            '<vim25:pathSet>info.state</vim25:pathSet>'
            '<vim25:pathSet>info.error</vim25:pathSet>'
            '</vim25:propSet>'
            '<vim25:objectSet>'
            '<vim25:obj type="Task">{task}</vim25:obj>'
            '</vim25:objectSet>'
            '</vim25:specSet>'
            '</vim25:RetrieveProperties>'
        ).format(task=task_moref)

        for attempt in range(1, max_attempts + 1):
            resp = self.http.post(soap_url, data=self._soap_envelope(query_body),
                                 headers=soap_headers, cookies=cookies, verify=self.verify_ssl)
            resp.raise_for_status()

            task_root = ET.fromstring(resp.text)
            state = None
            error_msg = None
            for elem in task_root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if tag == 'val' and elem.text in ('success', 'error', 'running', 'queued'):
                    state = elem.text
                elif tag == 'localizedMessage' and error_msg is None:
                    error_msg = elem.text

            if state == 'success':
                print('[OK] Tarea de snapshot completada')
                return
            elif state == 'error':
                raise Exception('Error en tarea de revert: {}'.format(
                    error_msg or 'sin detalle'))
            else:
                print('[*] Esperando tarea de revert... estado={} ({}/{})'.format(
                    state or '?', attempt, max_attempts))
                time.sleep(interval)

        raise Exception('Timeout esperando tarea de revert: {}'.format(task_moref))

    def wait_vm_power_state(self, vm_name=None, target_state='POWERED_OFF',
                            max_attempts=30, interval=10):
        """Esperar hasta que la VM alcance el estado de power indicado"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))

        for attempt in range(1, max_attempts + 1):
            state = self.get_vm_power_state(vm_id)
            if state == target_state:
                print('[OK] VM en estado {}'.format(target_state))
                return True
            print('[*] Esperando {} (actual: {}, intento {}/{})'.format(
                target_state, state, attempt, max_attempts))
            time.sleep(interval)

        raise Exception('Timeout esperando estado {} en VM {}'.format(target_state, vm_name))

    def connect_cdrom(self, vm_name=None):
        """Conectar el CD-ROM (para VMs encendidas)"""
        vm_name = vm_name or self.config.get('vm_name')
        vm = self.get_vm(vm_name)
        vm_id = vm.get('vm', vm.get('value', {}).get('vm'))
        
        cdroms = self.get_cdrom_devices(vm_id)
        if not cdroms:
            raise Exception('No se encontró dispositivo CD-ROM')
        
        cdrom = cdroms[0]
        cdrom_id = cdrom.get('cdrom', cdrom.get('key'))
        
        url = '{}/api/vcenter/vm/{}/hardware/cdrom/{}'.format(
            self.base_url, vm_id, cdrom_id
        )
        
        payload = {'connected': True}
        
        response = self.http.post(
            url + '?action=connect',
            headers=self._get_headers(),
            verify=self.verify_ssl
        )
        
        print('[OK] CD-ROM conectado')
        return True


def load_dotenv(config_path):
    """Cargar .env si las credenciales de vCenter no están ya en el entorno"""
    if os.environ.get('VCENTER_USER') and os.environ.get('VCENTER_PASSWORD'):
        return  # Ya cargadas por systemd u otro medio

    config_dir = os.path.dirname(os.path.abspath(config_path))
    env_path = os.path.join(config_dir, '.env')
    if not os.path.isfile(env_path):
        return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_config(config_path):
    """Cargar configuración YAML"""
    load_dotenv(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expandir variables de entorno
    def expand_env(obj):
        if isinstance(obj, str):
            if obj.startswith('${') and obj.endswith('}'):
                var_name = obj[2:-1]
                return os.environ.get(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env(i) for i in obj]
        return obj
    
    return expand_env(config)


def main():
    if len(sys.argv) < 3:
        print('Uso: vcenter_api.py <config_path> <action> [args...]')
        print('Acciones:')
        print('  upload_iso <local_iso_path>      - Subir ISO al datastore')
        print('  configure_cdrom [iso_path]       - Configurar CD-ROM de la VM con ISO')
        print('  power_on                         - Encender la VM')
        print('  power_off                        - Apagar la VM')
        print('  get_vm_status                    - Estado de la VM')
        print('  revert_snapshot [snapshot_name]  - Revertir al snapshot')
        print('  wait_powered_off                 - Esperar POWERED_OFF')
        print('  wait_powered_on                  - Esperar POWERED_ON')
        sys.exit(1)
    
    config_path = sys.argv[1]
    action = sys.argv[2]
    
    config = load_config(config_path)
    client = VCenterRESTClient(config)
    
    try:
        client.connect()
        
        if action == 'upload_iso':
            if len(sys.argv) < 4:
                print('Error: Falta ruta del ISO local')
                sys.exit(1)
            local_path = sys.argv[3]
            remote_path = client.upload_iso_to_datastore(local_path)
            # Imprimir el path remoto para que el script padre lo capture
            print('[REMOTE_ISO_PATH] {}'.format(remote_path))
            
        elif action == 'configure_cdrom':
            # Aceptar iso_path opcional como argumento
            iso_path = sys.argv[3] if len(sys.argv) > 3 else None
            client.configure_cdrom(iso_path=iso_path)
            
        elif action == 'power_on':
            client.power_on_vm()
            
        elif action == 'power_off':
            client.power_off_vm()
            
        elif action == 'get_vm_status':
            vm = client.get_vm(config['vcenter']['vm_name'])
            print('VM: {}'.format(json.dumps(vm, indent=2)))

        elif action == 'revert_snapshot':
            snapshot_name = sys.argv[3] if len(sys.argv) > 3 else None
            client.revert_to_snapshot(snapshot_name=snapshot_name)

        elif action == 'wait_powered_off':
            client.wait_vm_power_state(target_state='POWERED_OFF')

        elif action == 'wait_powered_on':
            client.wait_vm_power_state(target_state='POWERED_ON')

        else:
            print('Acción no reconocida: {}'.format(action))
            sys.exit(1)
            
    except Exception as e:
        print('[ERROR] {}'.format(str(e)))
        sys.exit(1)
        
    finally:
        client.disconnect()


if __name__ == '__main__':
    main()
