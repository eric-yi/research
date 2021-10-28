package com.icbc.devops.websocket.engine.applciation.archive.web;

import java.util.List;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.icbc.devops.websocket.engine.applciation.archive.service.*;

@RestController
class WebSocketController {

    @Autoware
    private websocketService websocketService;

  // Aggregate root
  // tag::get-aggregate-root[]
  @GetMapping("/websockets")
  List<websocket> all() {
    return websocketService.getAll();
  }
  // end::get-aggregate-root[]

  @PostMapping("/websockets")
  websocket newEmployee(@RequestBody websocket websocket) {
    return websocketService.save(websocket);
  }

  // Single item
  
  @GetMapping("/websockets/{id}")
  websocket one(@PathVariable Long id) {
    return websocketService.get(id);
  }

  @PutMapping("/websockets/{id}")
  websocket replaceEmployee(@RequestBody websocket websocket, @PathVariable Long id) {
    return websocketService.update(id);
  }

  @DeleteMapping("/websockets/{id}")
  void deleteEmployee(@PathVariable Long id) {
    websocketService.delete(id);
  }
}